import torch 


#  # self.sample_y_.shape (100, 62)
# temp = torch.linspace(0,9,10).reshape(-1,1)
# temp_y = temp.repeat(10,1)

# sample_y_ = torch.zeros((100, 10))
# sample_y_.scatter_(1, temp_y.type(torch.LongTensor), 1)
# # self.test_labels = self.data_Y[7*self.batch_size: 8*self.batch_size]


fill = torch.zeros([3, 3, 2, 2])
for i in range(3):
    fill[i, i, :, :] = 1

tmp = torch.Tensor([0,2,1]).unsqueeze(1)
y_vec_ = torch.zeros((3,3)).scatter_(1, tmp.type(torch.LongTensor), 1)
y_fill_ = fill[torch.max(y_vec_, 1)[1].squeeze()]

print('Here')