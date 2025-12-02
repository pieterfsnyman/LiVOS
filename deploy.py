import torch
import torch.nn as nn

from livos.model.livos_wrapper import LIVOS
from livos.utils.tensor_utils import aggregate


class InferenceInitializer(nn.Module):
    def __init__(self, network: LIVOS) -> None:
        super().__init__()
        self.network = network

    def forward(self, image_BCHW, mask_HW, objects):
        # Enocde image.
        ms_feats = self.network.encode_image(image_BCHW) # list of [B,C,H,W] tensors
        feat_BCHW = ms_feats[0] # 1/16
        pixfeat_BCHW = self.network.pix_projector(feat_BCHW) # 1/16
        key_BCHW = self.network.key_projector(feat_BCHW) # 1/16

        # Normalize the key.
        key_max_B1HW = torch.max(key_BCHW, dim=1, keepdim=True).values
        key_BCHW = (key_BCHW - key_max_B1HW).softmax(dim=1)

        B, _, H, W = key_BCHW.shape

        # The first frame must have mask and valid object(s).
        assert objects is not None and mask_HW is not None

        mask_NHW = torch.stack(
            [mask_HW == objects[i] for i in range(len(objects))], dim=0).float()            
        prob_with_bg_NHW = aggregate(mask_NHW, dim=0)

        # Initialize sensory.
        N = mask_NHW.shape[0]
        C = self.network.sensory_dim
        sensory_BNCHW = torch.zeros(B, N, C, H, W, device=image_BCHW.device)

        # Encode mask.
        mask_BNHW = mask_NHW.unsqueeze(0)
        value_BNCHW, sensory_BNCHW, obj_mem_BNQC = self.network.encode_mask(
            image_BCHW, pixfeat_BCHW, mask_BNHW, sensory_BNCHW, deep_update=True)
        
        # Initialize the object memory sum
        obj_mem_sum_BNQC = obj_mem_BNQC

        # Get the initial state.
        state_BNCC = torch.einsum('bkhw,bnvhw->bnkv', key_BCHW, value_BNCHW)
        
        key_sum_BCHW = key_BCHW
        sensory_BNCHW = sensory_BNCHW
        last_masks_BNHW = mask_BNHW
        
        return prob_with_bg_NHW, state_BNCC, key_sum_BCHW, sensory_BNCHW, obj_mem_sum_BNQC, last_masks_BNHW


class InferenceUpdater(nn.Module):
    def __init__(self, network: LIVOS) -> None:
        super().__init__()
        self.network = network

    def forward(self, image_BCHW, state_BNCC, key_sum_BCHW, sensory_BNCHW, obj_mem_sum_BNQC, last_masks_BNHW):
        # Enocde image.
        ms_feats = self.network.encode_image(image_BCHW) # list of [B,C,H,W] tensors
        feat_BCHW = ms_feats[0] # 1/16
        pixfeat_BCHW = self.network.pix_projector(feat_BCHW) # 1/16
        key_BCHW = self.network.key_projector(feat_BCHW) # 1/16        
        gate_BC = self.network.gate_projector(feat_BCHW) # 1/16

        # Normalize the key.
        key_max_B1HW = torch.max(key_BCHW, dim=1, keepdim=True).values
        key_BCHW = (key_BCHW - key_max_B1HW).softmax(dim=1)

        B, _, H, W = key_BCHW.shape
            
        # Get the value for the query frame.
        readout_BNCHW = torch.einsum(
            'bkhw,bnkv->bnvhw', key_BCHW, state_BNCC)
        norm_B = torch.einsum('bchw,bchw->b', key_BCHW, key_sum_BCHW)
        norm_B = norm_B.view(B, 1, 1, 1, 1)

        # Normalization for query readout.
        readout_BNCHW /= norm_B

        _, prob_with_bg_BNHW, sensory_BNCHW, _ = self.network.segment(
            ms_feats, readout_BNCHW, pixfeat_BCHW, last_masks_BNHW, 
            sensory_BNCHW, obj_mem_sum_BNQC, update_sensory=True)

        # Update the key sum.
        key_sum_BCHW = key_sum_BCHW + key_BCHW

        # Output probability map.
        prob_with_bg_NHW = prob_with_bg_BNHW[0]

        # Encode the mask to obtain new value.
        mask_BNHW = prob_with_bg_BNHW[:, 1:]
        value_BNCHW, sensory_BNCHW, obj_mem_BNQC = self.network.encode_mask(
            image_BCHW, pixfeat_BCHW, mask_BNHW, sensory_BNCHW, 
            deep_update=True)

        # Update state with a gate.
        state_BNCC = torch.einsum(
            'bvv,bnkv->bnkv', torch.diag_embed(gate_BC), state_BNCC)
        this_state_BNCC = torch.einsum(
            'bkhw,bnvhw->bnkv', key_BCHW, value_BNCHW)
        state_BNCC += this_state_BNCC

        # Update last masks.
        last_masks_BNHW = mask_BNHW

        # Update the object memory sum
        obj_mem_sum_BNQC = obj_mem_sum_BNQC + obj_mem_BNQC

        return prob_with_bg_NHW, state_BNCC, key_sum_BCHW, sensory_BNCHW, obj_mem_sum_BNQC, last_masks_BNHW


def main():
    with torch.inference_mode():
        network = LIVOS(model_type='base')
        network.load_weights(torch.load('weights/livos-wmose-480p.pth', weights_only=True))

        initializer = torch.jit.script(InferenceInitializer(network).cuda().eval())
        updater = torch.jit.script(InferenceUpdater(network).cuda().eval())

        initializer = torch.jit.optimize_for_inference(initializer)
        updater = torch.jit.optimize_for_inference(updater)

        initializer.save('weights/livos-wmose-480p-initializer.pth')
        updater.save('weights/livos-wmose-480p-updater.pth')


if __name__ == '__main__':
    main()
