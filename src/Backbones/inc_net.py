import copy

import timm
import torch
from Backbones.prompt import CodaPrompt
from torch import nn
from torchvision import transforms


def get_backbone(args):
    name = args.backbone.lower()
    if "_l2p" in name:
        if "l2p" in args.model:
            from Backbones import vision_transformer_l2p

            model = timm.create_model(
                args.backbone,
                pretrained=args.pretrained,
                num_classes=args.n_class,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                prompt_length=args.length,
                embedding_key=args.embedding_key,
                prompt_init=args.prompt_key_init,
                prompt_pool=args.prompt_pool,
                prompt_key=args.prompt_key,
                pool_size=args.size,
                top_k=args.top_k,
                batchwise_prompt=args.batchwise_prompt,
                prompt_key_init=args.prompt_key_init,
                head_type=args.head_type,
                use_prompt_mask=args.use_prompt_mask,
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    elif "_dualprompt" in name:
        if "dualprompt" in args.model:
            from Backbones import vision_transformer_dual_prompt

            model = timm.create_model(
                args.backbone,
                pretrained=args.pretrained,
                num_classes=args.n_class,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                prompt_length=args.length,
                embedding_key=args.embedding_key,
                prompt_init=args.prompt_key_init,
                prompt_pool=args.prompt_pool,
                prompt_key=args.prompt_key,
                pool_size=args.n_task,
                top_k=args.top_k,
                batchwise_prompt=args.batchwise_prompt,
                prompt_key_init=args.prompt_key_init,
                head_type=args.head_type,
                use_prompt_mask=args.use_prompt_mask,
                use_g_prompt=args.use_g_prompt,
                g_prompt_length=args.g_prompt_length,
                g_prompt_layer_idx=args.g_prompt_layer_idx,
                use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
                use_e_prompt=args.use_e_prompt,
                e_prompt_layer_idx=args.e_prompt_layer_idx,
                use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
                same_key_value=args.same_key_value,
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    elif "_coda_prompt" in name:
        if "coda_prompt" in args.model:
            from Backbones import vision_transformer_coda_prompt

            model = timm.create_model(args.backbone, pretrained=args.pretrained)
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    elif "_pmae" in name:
        if "pmae" in args.model:
            from Backbones import vision_transformer_pmae

            model = timm.create_model(args.backbone, pretrained=True)
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class PromptVitNet(nn.Module):
    def __init__(self, args):
        super(PromptVitNet, self).__init__()
        self.backbone = get_backbone(args)

        if args.get_original_backbone:
            self.original_backbone = self.get_original_backbone(args)
        else:
            self.original_backbone = None

        for p in self.original_backbone.parameters():
            p.requires_grad = False

        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in self.backbone.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

        if args.w_pmae:
            pmae_args = copy.deepcopy(args)
            pmae_args.backbone = args.pmae_backbone
            self.pmae = PromptMAE_minimal(pmae_args)

    def get_original_backbone(self, args):
        return timm.create_model(
            args.backbone,
            pretrained=args.pretrained,
            num_classes=args.n_class,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        ).eval()

    def forward(self, x, task_id=-1, train=False):
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features = self.original_backbone(x)["pre_logits"]
            else:
                cls_features = None

        x = self.backbone(x, task_id=task_id, cls_features=cls_features, train=train)
        return x


class CodaPromptVitNet(nn.Module):
    def __init__(self, args):
        super(CodaPromptVitNet, self).__init__()
        self.args = args
        self.backbone = get_backbone(args)
        self.feature_dim = self.backbone.embed_dim
        self.fc = nn.Linear(self.feature_dim, args.n_class)
        self.prompt = CodaPrompt(
            self.feature_dim, args.n_task, args.prompt_param, self.feature_dim
        )

        if args.w_pmae:
            pmae_args = copy.deepcopy(args)
            pmae_args.backbone = args.pmae_backbone
            self.pmae = PromptMAE_minimal(pmae_args)

    # pen: get penultimate features
    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.backbone(x)
                q = q[:, 0, :]
            out, prompt_loss = self.backbone(x, prompt=self.prompt, q=q, train=train)
            out = out[:, 0, :]
        else:
            out, _ = self.backbone(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.fc(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out


class PromptMAE(nn.Module):
    def __init__(self, args):
        super(PromptMAE, self).__init__()
        self.args = args
        self.backbone = get_backbone(args)
        self.encoder_feature_dim = self.backbone.encoder.embed_dim
        self.decoder_feature_dim = self.backbone.decoder.decoder_embed_dim
        self.fc = nn.Linear(self.encoder_feature_dim, args.n_class)
        self.prompt_cls = nn.Parameter(torch.randn((5, 20, self.encoder_feature_dim)))
        nn.init.uniform_(self.prompt_cls, -1, 1)

        self.prompt_mae = nn.Parameter(torch.randn((1, 5, self.decoder_feature_dim)))
        nn.init.uniform_(self.prompt_mae, -1, 1)

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    def forward(self, x, mask_ratio=0):
        encoder_prompt = self.prompt_cls.unsqueeze(0)
        encoder_prompt = encoder_prompt.expand(x.shape[0], -1, -1, -1)
        features, mask, ids_restore = self.backbone.forward_encoder(
            x, mask_ratio=mask_ratio, encoder_prompt=encoder_prompt
        )
        output = self.fc(features[:, 0])
        return output

    def forward_encoder(self, x, mask_ratio=0.75):
        latent, mask, ids_restore = self.backbone.forward_encoder(
            x, mask_ratio=mask_ratio
        )
        return latent, mask, ids_restore

    def forward_decoder(self, x, ids_restore, task_decoder_prompt=None):
        decoder_prompt = (
            self.prompt_mae.unsqueeze(0)
            if task_decoder_prompt is None
            else task_decoder_prompt.unsqueeze(0)
        )
        decoder_prompt = decoder_prompt.expand(x.shape[0], -1, -1, -1)

        if self.args.wo_prompt_mae:
            decoder_prompt = None

        y = self.backbone.forward_decoder(x, ids_restore, decoder_prompt=decoder_prompt)
        return y

    def forward_mae(self, x, mask_ratio=0.75):
        decoder_prompt = self.prompt_mae.unsqueeze(0)
        decoder_prompt = decoder_prompt.expand(x.shape[0], -1, -1, -1)

        if self.args.wo_prompt_mae:
            decoder_prompt = None

        loss, pred, mask = self.backbone(
            x,
            mask_ratio=mask_ratio,
            decoder_prompt=decoder_prompt,
        )
        return loss, pred, mask

    def unpatchify(self, y):
        images = self.backbone.unpatchify(y)
        return images


class PromptMAE_minimal(nn.Module):
    def __init__(self, args):
        super(PromptMAE_minimal, self).__init__()
        self.args = args
        self.backbone = get_backbone(args)
        self.decoder_feature_dim = self.backbone.decoder.decoder_embed_dim

        self.prompt_mae = nn.Parameter(torch.randn((1, 5, self.decoder_feature_dim)))
        nn.init.uniform_(self.prompt_mae, -1, 1)

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    def forward_encoder(self, x, mask_ratio=0.75):
        latent, mask, ids_restore = self.backbone.forward_encoder(
            x, mask_ratio=mask_ratio
        )
        return latent, mask, ids_restore

    def forward_decoder(self, x, ids_restore, task_decoder_prompt=None):
        decoder_prompt = (
            self.prompt_mae.unsqueeze(0)
            if task_decoder_prompt is None
            else task_decoder_prompt.unsqueeze(0)
        )
        decoder_prompt = decoder_prompt.expand(x.shape[0], -1, -1, -1)

        if self.args.wo_prompt_mae:
            decoder_prompt = None

        y = self.backbone.forward_decoder(x, ids_restore, decoder_prompt=decoder_prompt)
        return y

    def forward_mae(self, x, mask_ratio=0.75):
        decoder_prompt = self.prompt_mae.unsqueeze(0)
        decoder_prompt = decoder_prompt.expand(x.shape[0], -1, -1, -1)

        if self.args.wo_prompt_mae:
            decoder_prompt = None

        loss, pred, mask = self.backbone(
            x,
            mask_ratio=mask_ratio,
            decoder_prompt=decoder_prompt,
        )
        return loss, pred, mask

    def unpatchify(self, y):
        images = self.backbone.unpatchify(y)
        return images
