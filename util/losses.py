import torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from models.clip_relevancy import ClipRelevancy
from util.aug_utils import RandomSizeCrop
from util.util import get_screen_template, get_text_criterion, get_augmentations_template
import torch.nn as nn


class L_exp(nn.Module):

    def __init__(self, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mean_val = mean_val

    def forward(self, x):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k

class LossG(torch.nn.Module):
    def __init__(self, cfg, clip_extractor):
        super().__init__()

        self.cfg = cfg

        # calculate target text embeddings
        template = get_augmentations_template()
        self.src_e = clip_extractor.get_text_embedding(cfg["src_text"], template)
        self.target_comp_e = clip_extractor.get_text_embedding("a scene " + cfg["comp_text"], template)
        self.origin_comp_e = clip_extractor.get_text_embedding("a normal scene", template)

        self.target_greenscreen_e = clip_extractor.get_text_embedding(cfg["screen_text"], get_screen_template())

        self.clip_extractor = clip_extractor
        self.text_criterion = get_text_criterion(cfg)

        if cfg["bootstrap_epoch"] > 0 and cfg["lambda_bootstrap"] > 0:
            self.relevancy_extractor = ClipRelevancy(cfg)
            self.relevancy_criterion = torch.nn.MSELoss()
            self.lambda_bootstrap = cfg["lambda_bootstrap"]

        self.Lexp = L_exp(0.6) #Exposure Control Loss 调整亮度
        self.Lsa = Sa_Loss() #Color Constancy Loss 色彩一致性
    def forward(self, outputs, inputs):
        losses = {}
        loss_G = 0

        all_outputs_composite = []
        all_outputs_greenscreen = []
        all_greenscreen = []
        all_outputs_edit = []
        all_outputs_alpha = []
        all_inputs = []
        all_illu_inputs = []
        all_outputs_att = []
        all_outputs_illu = []
        all_outputs_comp_R = []
        for out, ins in zip(["output_crop", "output_image"], ["input_crop", "input_image"]):
            if out not in outputs:
                continue
            all_outputs_composite += outputs[out]["composite"]
            all_outputs_greenscreen += outputs[out]["edit_on_greenscreen"]
            all_outputs_edit += outputs[out]["edit"]
            all_outputs_alpha += outputs[out]["alpha"]
            all_outputs_att += outputs[out]["att"]
            all_outputs_illu += outputs[out]["illu"]
            all_outputs_comp_R += outputs[out]["composite_R"]
            all_greenscreen += outputs[out]["greenscreen"]
            all_inputs += inputs[ins]
            all_illu_inputs += inputs["illu_"+ins]

        # calculate alpha bootstrapping loss
        if inputs["step"] < self.cfg["bootstrap_epoch"] and self.cfg["lambda_bootstrap"] > 0:
            losses["loss_bootstrap"] = self.calculate_relevancy_loss(all_outputs_alpha, all_inputs)

            if self.cfg["bootstrap_scheduler"] == "linear":
                lambda_bootstrap = self.cfg["lambda_bootstrap"] * (
                    1 - (inputs["step"] + 1) / self.cfg["bootstrap_epoch"]
                )
            elif self.cfg["bootstrap_scheduler"] == "exponential":
                lambda_bootstrap = self.lambda_bootstrap * 0.99
                self.lambda_bootstrap = lambda_bootstrap
            elif self.cfg["bootstrap_scheduler"] == "none":
                lambda_bootstrap = self.lambda_bootstrap
            else:
                raise ValueError("Unknown bootstrap scheduler")
            lambda_bootstrap = max(lambda_bootstrap, self.cfg["lambda_bootstrap_min"])
            loss_G += losses["loss_bootstrap"] * lambda_bootstrap

        # calculate structure loss
        if self.cfg["lambda_structure"] > 0:
            losses["loss_structure"] = self.calculate_structure_loss(all_outputs_composite, all_inputs)
            loss_G += losses["loss_structure"] * self.cfg["lambda_structure"]

        # calculate composition loss
        if self.cfg["lambda_composition"] > 0:
            losses["loss_comp_clip"] = self.calculate_clip_loss(all_outputs_composite, self.target_comp_e-self.origin_comp_e, all_inputs)

            losses["loss_comp_dir"] = self.calculate_clip_dir_loss(
                all_inputs, all_outputs_composite, self.target_comp_e-self.origin_comp_e
            )
            # losses["loss_comp_dir"] = 0
            loss_G += (losses["loss_comp_clip"] + losses["loss_comp_dir"]) * self.cfg["lambda_composition"]

        # calculate sparsity loss
        if self.cfg["lambda_sparsity"] > 0:
            total, l0, l1 = self.calculate_alpha_reg(all_outputs_alpha)
            losses["loss_sparsity"] = total
            losses["loss_sparsity_l0"] = l0
            losses["loss_sparsity_l1"] = l1

            loss_G += losses["loss_sparsity"] * self.cfg["lambda_sparsity"]

        # calculate screen loss
        # if self.cfg["lambda_screen"] > 0:
        #
        #     losses["loss_screen"] = self.calculate_clip_loss(all_outputs_greenscreen, self.target_greenscreen_e, all_greenscreen)
        #     loss_G += losses["loss_screen"] * self.cfg["lambda_screen"]

        # calculate illunination smooth loss
        if self.cfg["lambda_illu_smooth"] > 0:
            losses["loss_smooth"] = self.calculate_illumination_smooth_loss(all_outputs_att, all_inputs) \
                                    + self.calculate_illumination_smooth_MSE_loss(all_outputs_illu, all_illu_inputs)
            loss_G += losses["loss_smooth"]*self.cfg["lambda_illu_smooth"]

        # calculate illumination loss
        if self.cfg["lambda_illu"] > 0:
            losses["loss_illu"] = self.calculate_illumination_loss(all_outputs_composite)
            loss_G += losses["loss_illu"] * self.cfg["lambda_illu"]

        # calculate color loss
        # if self.cfg["lambda_color"] > 0:
        #     losses["loss_color"] = self.calculate_color_loss(all_outputs_composite)
        #     loss_G += losses["loss_color"] * self.cfg["lambda_color"]


        losses["loss"] = loss_G
        return losses

    def calculate_alpha_reg(self, prediction):
        """
        Calculate the alpha sparsity term: linear combination between L1 and pseudo L0 penalties
        """
        l1_loss = 0.0
        for el in prediction:
            l1_loss += el.mean()
        l1_loss = l1_loss / len(prediction)
        loss = self.cfg["lambda_alpha_l1"] * l1_loss
        # Pseudo L0 loss using a squished sigmoid curve.
        l0_loss = 0.0
        for el in prediction:
            l0_loss += torch.mean((torch.sigmoid(el * 5.0) - 0.5) * 2.0)
        l0_loss = l0_loss / len(prediction)
        loss += self.cfg["lambda_alpha_l0"] * l0_loss
        return loss, l0_loss, l1_loss

    def calculate_clip_loss(self, outputs, target_embeddings, inputs):
        # randomly select embeddings
        n_embeddings = np.random.randint(1, len(target_embeddings) + 1)
        target_embeddings = target_embeddings[torch.randint(len(target_embeddings), (n_embeddings,))]

        loss = 0.0
        for img,input in zip(outputs,inputs):  # avoid memory limitations
            img_e = self.clip_extractor.get_image_embedding(img.unsqueeze(0))
            img_i = self.clip_extractor.get_image_embedding(input.unsqueeze(0))
            for target_embedding in target_embeddings:
                loss += self.text_criterion(img_e, img_i + 1.75*target_embedding.unsqueeze(0))

        loss /= len(outputs) * len(target_embeddings)
        return loss

    def calculate_clip_dir_loss(self, inputs, outputs, target_embeddings):
        # randomly select embeddings
        n_embeddings = np.random.randint(1, min(len(self.src_e), len(target_embeddings)) + 1)
        idx = torch.randint(min(len(self.src_e), len(target_embeddings)), (n_embeddings,))
        target_embeddings = target_embeddings[idx]
        target_dirs = target_embeddings

        loss = 0.0
        for in_img, out_img in zip(inputs, outputs):  # avoid memory limitations
            in_e = self.clip_extractor.get_image_embedding(in_img.unsqueeze(0))
            out_e = self.clip_extractor.get_image_embedding(out_img.unsqueeze(0))
            for target_dir in target_dirs:
                loss += 1 - torch.nn.CosineSimilarity()(out_e - in_e, target_dir.unsqueeze(0)).mean()

        loss /= len(outputs) * len(target_dirs)
        return loss

    def calculate_structure_loss(self, outputs, inputs):
        loss = 0.0
        for input, output in zip(inputs, outputs):
            with torch.no_grad():
                target_self_sim = self.clip_extractor.get_self_sim(input.unsqueeze(0))
            current_self_sim = self.clip_extractor.get_self_sim(output.unsqueeze(0))
            loss = loss + torch.nn.MSELoss()(current_self_sim, target_self_sim)
        loss = loss / len(outputs)
        return loss

    def calculate_relevancy_loss(self, alpha, input_img):
        positive_relevance_loss = 0.0
        for curr_alpha, curr_img in zip(alpha, input_img):
            x = torch.stack([curr_alpha, curr_img], dim=0)  # [2, 3, H, W]
            x = T.Compose(
                [
                    RandomSizeCrop(min_cover=self.cfg["bootstrapping_min_cover"]),
                    T.Resize((224, 224)),
                ]
            )(x)
            curr_alpha, curr_img = x[0].unsqueeze(0), x[1].unsqueeze(0)
            positive_relevance = self.relevancy_extractor(curr_img)
            positive_relevance_loss = self.relevancy_criterion(curr_alpha[0], positive_relevance.repeat(3, 1, 1))
            if self.cfg["use_negative_bootstrap"]:
                negative_relevance = self.relevancy_extractor(curr_img, negative=True)
                relevant_values = negative_relevance > self.cfg["bootstrap_negative_map_threshold"]
                negative_alpha_local = (1 - curr_alpha) * relevant_values.unsqueeze(1)
                # negative_alpha_local = curr_alpha * relevant_values.unsqueeze(1)
                negative_relevance_local = negative_relevance * relevant_values
                negative_relevance_loss = self.relevancy_criterion(
                    negative_alpha_local,
                    negative_relevance_local.unsqueeze(1).repeat(1, 3, 1, 1),
                )
                positive_relevance_loss += negative_relevance_loss
        positive_relevance_loss = positive_relevance_loss / len(alpha)
        return positive_relevance_loss

    def calculate_illumination_smooth_loss(self, illu_outputs, input_img):
        def gradient(input_tensor, direction):
            input_tensor = input_tensor.permute(0, 3, 1, 2)

            smooth_kernel_x = torch.reshape(torch.Tensor([[0., 0.], [-1., 1.]]), (1, 1, 2, 2)).cuda()
            smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)

            if direction == "x":
                kernel = smooth_kernel_x
            else:
                kernel = smooth_kernel_y

            out = F.conv2d(input_tensor, kernel, padding=(1, 1)).cuda()
            out = torch.abs(out[:, :, 0:input_tensor.size()[2], 0:input_tensor.size()[3]])

            return out.permute(0, 2, 3, 1)

        def ave_gradient(input_tensor, direction):
            return (F.avg_pool2d(gradient(input_tensor, direction).permute(0, 3, 1, 2).cuda(), 3, stride=1, padding=1)) \
                .permute(0, 2, 3, 1)

        def smooth(input_l, input_r):
            input_l = input_l.permute(1,2,0).unsqueeze(0)
            input_r = input_r.permute(1, 2, 0).unsqueeze(0)
            rgb_weights = torch.Tensor([0.2989, 0.5870, 0.1140]).cuda()
            input_r = torch.tensordot(input_r, rgb_weights, dims=([-1], [-1]))
            input_r = torch.unsqueeze(input_r, -1)
            input_l = torch.tensordot(input_l, rgb_weights, dims=([-1], [-1]))
            input_l = torch.unsqueeze(input_l, -1)
            return torch.mean(
                gradient(input_l, 'x') * torch.exp(-10 * ave_gradient(input_r, 'x')) +
                gradient(input_l, 'y') * torch.exp(-10 * ave_gradient(input_r, 'y'))
            )

        loss = 0.0
        for input, output in zip(input_img, illu_outputs):
            ismooth_loss = smooth(output, input)

            loss = loss + ismooth_loss
        loss = loss / len(illu_outputs)
        return loss


    def calculate_illumination_smooth_MSE_loss(self, illu_outputs, input_img):

        loss = 0.0
        MSELoss = nn.MSELoss()
        for input, output in zip(input_img, illu_outputs):
            output_grey = output.sum(axis=1,keepdim=True)
            ismooth_loss = MSELoss(output_grey, input)

            loss = loss + ismooth_loss
        loss = loss / len(illu_outputs)
        return loss

    def calculate_illumination_loss(self, illu_outputs):

        loss = 0.0
        for output in illu_outputs:
            illu_loss = self.Lexp(output)
            loss = loss + illu_loss
        loss = loss / len(illu_outputs)
        return loss

    def calculate_color_loss(self, illu_outputs):

        loss = 0.0
        for output in illu_outputs:
            color_loss = self.Lsa(output.unsqueeze(0))
            loss = loss + color_loss
        loss = loss / len(illu_outputs)
        return loss