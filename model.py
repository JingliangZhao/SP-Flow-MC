import torch
import torch.nn as nn
import torch.nn.functional as F


# Model for flow vector estimation. The outputs are alhpa and beta.
class Flow_vector_model(nn.Module):
    
    # Loss function: nn.MSELoss()
    
    def __init__(self, in_channel=3, out_channel=2, training=True):
        super(Flow_vector_model, self).__init__()
        
        self.out_ch = out_channel
        self.training = training
        input_ch = in_channel;   base_ch = 16
        
        self.C1 = nn.Sequential(
            # c1
            nn.Conv2d(input_ch, base_ch, kernel_size=5, padding='same'),       
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            
            nn.Conv2d(base_ch, base_ch, kernel_size=5, padding='same'),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
         )
         
        self.C2 = nn.Sequential(
            # c2
            nn.Conv2d(base_ch, base_ch*2, kernel_size=4, padding='same'),       
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(),
            
            nn.Conv2d(base_ch*2, base_ch*2, kernel_size=4, padding='same'),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.C3 = nn.Sequential(
            # c3
            nn.Conv2d(base_ch*2, base_ch*4, kernel_size=3, padding='same'),       
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(),
            
            nn.Conv2d(base_ch*4, base_ch*4, kernel_size=3, padding='same'),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc1 = nn.Linear(base_ch*4 * 4*4, 200)
        self.fc2 = nn.Linear(200, self.out_ch)
    
    def forward(self, t0):
        
        t1 = self.C1(t0)                                        # [B, 32, 16, 16]
        t2 = F.dropout(self.C2(t1), 0.35, self.training)        # [B, 64, 8, 8]
        t3 = F.dropout(self.C3(t2), 0.35, self.training)        # [B, 128, 4, 4]
        
        f  = torch.flatten(t3, 1)                           # [B, 2048]
        
        f1 = F.dropout(self.fc1(f), 0.5, self.training)     # [B, 200]
        f2 = self.fc2(f1)
        
        out = torch.sigmoid(f2)
        
        return out



# Model for local vessel segmentation
class Segmentation_model(nn.Module):
    
    # Loss function: DiceLoss()
    # In training phase, loss_fin = loss4 + 0.4 * (loss1 + loss2 + loss3),
    # where loss1~loss4 are the DiceLoss of output1~output4, respectively.
    
    def __init__(self, in_channel=1, out_channel=2 ,training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(in_channel, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        self.map4 = nn.Sequential(
            nn.Conv3d(32, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),

            nn.Softmax(dim=1)
        )

        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        
        long_range1 = self.encoder_stage1(inputs) + inputs      # [1, 16, 64, 64, 64]
        short_range1 = self.down_conv1(long_range1)             # [1, 32, 32, 32, 32]

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.dorp_rate, self.training) # [1, 32, 32, 32, 32]
        short_range2 = self.down_conv2(long_range2)                         # [1, 64, 16, 16, 16]

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.dorp_rate, self.training) # [1, 64, 16, 16, 16]
        short_range3 = self.down_conv3(long_range3)                         # [1, 128, 8, 8, 8]

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.dorp_rate, self.training) # [1, 128, 8, 8, 8]
        short_range4 = self.down_conv4(long_range4)                         # [1, 256, 8, 8, 8]

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, self.dorp_rate, self.training)   # [1, 256, 8, 8, 8]

        output1 = self.map1(outputs)   # [1, 2, 64, 64, 64]

        short_range6 = self.up_conv2(outputs)   # [1, 128, 16, 16, 16]

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, self.dorp_rate, self.training)   # [1, 128, 16, 16, 16]

        output2 = self.map2(outputs)   # [1, 2, 64, 64, 64]

        short_range7 = self.up_conv3(outputs)   # [1, 64, 32, 32, 32]

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, self.dorp_rate, self.training)   # [1, 64, 32, 32, 32]

        output3 = self.map3(outputs)   # [1, 2, 64, 64, 64]

        short_range8 = self.up_conv4(outputs)   # [1, 32, 64, 64, 64]

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
        # [1, 32, 64, 64, 64]

        output4 = self.map4(outputs)   # [1, 2, 64, 64, 64]

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4



































#

#