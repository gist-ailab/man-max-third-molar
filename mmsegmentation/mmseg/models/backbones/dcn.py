import torch
import torch.nn as nn
import torchvision.ops


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], requires_grad=True))
        self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, requires_grad=True))
            
    def forward(self, x):
        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.weight, 
                                          bias=self.bias, 
                                          padding=self.padding,
                                          stride=self.stride,
                                          )
        return x


class DeformableConv2dV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2dV2, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], requires_grad=True))
        self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, requires_grad=True))
        
    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.weight, 
                                          bias=self.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class ConstrainedDeformableConv2d2DoF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(ConstrainedDeformableConv2d2DoF, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.kernel_size = kernel_size
        self.repeat = (1, 3, 1, 1)
        
        # kernel_size = (k_h, k_w)
        self.offset_conv = nn.Conv2d(in_channels,
                                    2,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], requires_grad=True))
        self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, requires_grad=True))

    def forward(self, x):

        o = self.offset_conv(x)
        o_0 = torch.zeros(size=(o.size(0), 1, o.size(2), o.size(3)), requires_grad=False).cuda()

        o_y, o_x = o[:, :1], o[:, 1:]
        offset_y = torch.cat((-o_y, o_0, o_y), dim=1).repeat(self.repeat)
        offset_x = torch.cat((-o_x.repeat(self.repeat), o_0.repeat(self.repeat), o_x.repeat(self.repeat)), dim=1)

        offset = torch.cat((offset_y, offset_x), dim=1)

        # modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.weight, 
                                          bias=self.bias, 
                                          padding=self.padding,
                                        #   mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class ConstrainedDeformableConv2d3DoF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(ConstrainedDeformableConv2d3DoF, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.kernel_size = kernel_size
        self.repeat = (1, 3, 1, 1)
        
        # kernel_size = (k_h, k_w)
        self.offset_conv = nn.Conv2d(in_channels,
                                    3,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], requires_grad=True))
        self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, requires_grad=True))
                                      

    def forward(self, x):

        o = self.offset_conv(x)
        o_0 = torch.zeros(size=(o.size(0), 1, o.size(2), o.size(3)), requires_grad=False).cuda()
        o_1 = torch.ones(size=(o.size(0), 1, o.size(2), o.size(3)), requires_grad=False).cuda()

        o_y, o_x, o_r = o[:, 0:1], o[:, 1:2], o[:, 2:3]

        o_yp1 = o_y + o_1
        o_xp1 = o_x + o_1

        s = torch.sin(o_r)
        c = torch.cos(o_r)

        offset_y = torch.cat(
            (
                - o_xp1 * s - o_yp1 * c + o_1,
                - o_xp1 * s,
                - o_xp1 * s + o_yp1 * c - o_1,
                - o_yp1 * c + o_1,
                o_0,
                o_yp1 * c - o_1,
                o_xp1 * s - o_yp1 * c  + o_1,
                o_xp1 * s,
                o_xp1 * s + o_yp1 * c - o_1
            ),
            dim=1
        )

        offset_x = torch.cat(
            (
                - o_xp1 * c - o_yp1 * (-s) + o_1,
                - o_xp1 * c + o_1,
                - o_xp1 * c + o_yp1 * (-s) + o_1,
                - o_yp1 * (-s),
                o_0,
                o_yp1 * (-s),
                o_xp1 * c - o_yp1 * (-s) - o_1,
                o_xp1 * c - o_1,
                o_xp1 * c + o_yp1 * (-s) - o_1
            ),
            dim=1
        )

        offset = torch.cat((offset_y, offset_x), dim=1)

        # modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.weight, 
                                          bias=self.bias, 
                                          padding=self.padding,
                                        #   mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class ConstrainedDeformableConv2d4DoF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(ConstrainedDeformableConv2d4DoF, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.kernel_size = kernel_size
        self.repeat = (1, 3, 1, 1)
        
        # kernel_size = (k_h, k_w)
        self.offset_conv = nn.Conv2d(in_channels,
                                    4,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        assert kernel_size[0] == 3 and kernel_size[1] == 3
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], requires_grad=True))
        self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, requires_grad=True))
                                      

    def forward(self, x):

        o = self.offset_conv(x)
        o_0 = torch.zeros(size=(o.size(0), 1, o.size(2), o.size(3)), requires_grad=False).cuda()

        o_t, o_b, o_l, o_r = o[:, 0:1], o[:, 1:2], o[:, 2:3], o[:, 3:4]

        offset_y = torch.cat((o_t, o_0, o_b), dim=1).repeat(self.repeat)
        offset_x = torch.cat((o_l.repeat(self.repeat), o_0.repeat(self.repeat), o_r.repeat(self.repeat)), dim=1)

        offset = torch.cat((offset_y, offset_x), dim=1)
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.weight, 
                                          bias=self.bias, 
                                          padding=self.padding,
                                        #   mask=modulator,
                                          stride=self.stride,
                                          )
        return x