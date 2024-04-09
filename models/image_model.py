import torch
from .networks import define_G

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.netG = define_G(cfg).to(device)

    def render(self, net_output, bg_image=None):
        assert net_output.min() >= 0 and net_output.max() <= 1
        edit = net_output[:, :3]
        alpha = net_output[:, 3].unsqueeze(1).repeat(1, 3, 1, 1)
        att = net_output[:, 4].unsqueeze(1).repeat(1, 3, 1, 1)
        illumination = net_output[:,5:8]
        greenscreen = torch.zeros_like(edit).to(edit.device)
        greenscreen[:, 1, :, :] = 177 / 255
        greenscreen[:, 2, :, :] = 64 / 255
        edit_on_greenscreen = alpha*edit + (1-alpha)*greenscreen
        outputs = {"edit": edit,"illu":illumination, "alpha": alpha,"att":att, "edit_on_greenscreen": edit_on_greenscreen, "greenscreen": greenscreen}
        # print(net_output.size())
        # print(alpha.size())
        # print(edit.size())
        # print(bg_image.size())
        # print(illumination.size())
        if bg_image is not None:
            outputs["composite"] = (edit*alpha + (1-alpha)*illumination)*bg_image*(4*att)
            outputs["composite_R"] = (edit*alpha + (1-alpha)*illumination)*bg_image
        return outputs

    def forward(self, input):
        outputs = {}
        # print(input["input_crop"].size())
        # augmented examples
        if "input_crop" in input:
            outputs["output_crop"] = self.render(self.netG(input["input_crop"]), bg_image=input["input_crop"])

        # pass the entire image (w/o augmentations)
        if "input_image" in input:
            outputs["output_image"] = self.render(self.netG(input["input_image"]), bg_image=input["input_image"])

        # move outputs to list
        for outer_key in outputs.keys():
            for key, value in outputs[outer_key].items():
                outputs[outer_key][key] = [value[0]]

        return outputs

