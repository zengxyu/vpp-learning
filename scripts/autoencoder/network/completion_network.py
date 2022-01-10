import torch
import torch.nn as nn
import torch.utils.data
import MinkowskiEngine as ME


class CompletionShadowNet(nn.Module):
    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self):
        nn.Module.__init__(self)

        # Input sparse tensor must have tensor stride 128.
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(1, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
        )
        self.linear_mean = ME.MinkowskiLinear(enc_ch[3], enc_ch[3], bias=True)
        self.linear_log_var = ME.MinkowskiLinear(enc_ch[3], enc_ch[3], bias=True)
        # Decoder

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(
            dec_ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[2],
                dec_ch[1],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(
            dec_ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(
            dec_ch[0], 1, kernel_size=1, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            # target_label = torch.zeros(len(out), dtype=torch.bool, device=out.device)

            cm = out.coordinate_manager
            # 把 target 跳着取， 得到跳着取的健值， 相当于降采样
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0],
            )
            out_coordinate_map_key = out.coordinate_map_key
            # construct mapping
            kernel_map = cm.kernel_map(
                out_coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            for k, curr_in in kernel_map.items():
                input_row_indices = curr_in[0].long()
                max_input = input_row_indices.max()
                output_row_indices = curr_in[1].long()
                target[input_row_indices] = 1
                # target_label[input_row_indices] = out[input_row_indices]
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, partial_in, target_key):
        out_cls, targets = [], []

        enc_s1 = self.enc_block_s1(partial_in)
        print("enc_s1:", len(enc_s1))
        enc_s2 = self.enc_block_s1s2(enc_s1)
        print("enc_s2:", len(enc_s2))

        enc_s4 = self.enc_block_s2s4(enc_s2)
        print("enc_s4:", len(enc_s4))

        enc_s8 = self.enc_block_s4s8(enc_s4)
        print("enc_s8:", len(enc_s8))

        dec_s4 = self.dec_block_s8s4(enc_s8)
        print("dec_s4:", len(dec_s4))

        # Add encoder features
        # dec_s4 = dec_s4 + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)

        target = self.get_target(dec_s4, target_key)
        targets.append(target)
        out_cls.append(dec_s4_cls)
        keep_s4 = (dec_s4_cls.F > 0).squeeze()

        if self.training:
            keep_s4 += target

        # Remove voxels s4
        dec_s4 = self.pruning(dec_s4, keep_s4)
        print("pruning dec_s4:", len(dec_s4))

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4)
        print("dec_s2:", len(dec_s2))

        # Add encoder features
        # dec_s2 = dec_s2 + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)

        target = self.get_target(dec_s2, target_key)
        targets.append(target)
        out_cls.append(dec_s2_cls)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()

        if self.training:
            keep_s2 += target

        # Remove voxels s2
        dec_s2 = self.pruning(dec_s2, keep_s2)
        print("pruning dec_s2:", len(dec_s2))

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)
        print("dec_s1:", len(dec_s1))

        dec_s1_cls = self.dec_s1_cls(dec_s1)

        # Add encoder features
        # dec_s1 = dec_s1 + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        target = self.get_target(dec_s1, target_key)
        targets.append(target)
        out_cls.append(dec_s1_cls)
        keep_s1 = (dec_s1_cls.F > 0).squeeze()

        # Last layer does not require adding the target
        # if self.training:
        #     keep_s1 += target

        # Remove voxels s1
        dec_s1 = self.pruning(dec_s1, keep_s1)
        print("pruning dec_s1:", len(dec_s1))

        return out_cls, targets, dec_s1
