

# python

## 1.pytorch 实现NMS

```python
from torch import Tensor
import torch

def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
 
 
def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)
 
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] # N中一个和M个比较； 所以由N，M 个
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
 
    wh = (rb - lt).clamp(min=0)  # [N,M,2]  #小于0的为0  clamp 钳；夹钳；
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  
 
    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # NxM， boxes1中每个框和boxes2中每个框的IoU值；
 
 
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float):
    """
    :param boxes: [N, 4]， 此处传进来的框，是经过筛选（NMS之前选取过得分TopK）之后， 在传入之前处理好的；
    :param scores: [N]
    :param iou_threshold: 0.7
    :return:
    """
    keep = []  # 最终保留的结果， 在boxes中对应的索引；
    idxs = scores.argsort()  # 值从小到大的 索引
    while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
        # 得分最大框对应的索引, 以及对应的坐标
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        keep.append(max_score_index)
        if idxs.size(0) == 1:  # 就剩余一个框了；
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]
 
    keep = idxs.new(keep)  # Tensor
    return keep
 
 
box =  torch.tensor([[2,3.1,7,5],[3,4,8,4.8],[4,4,5.6,7],[0.1,0,8,1]]) 
score = torch.tensor([0.5, 0.3, 0.2, 0.4])
 
output = nms(boxes=box, scores=score, iou_threshold=0.3)
print(output)

```

## 2.numpy 实现NMS

```python
import numpy as np
from numpy import array

def box_area(boxes :array):
    """
    :param boxes: [N, 4]
    :return: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(box1 :array, box2: array):
    """
    :param box1: [N, 4]
    :param box2: [M, 4]
    :return: [N, M]
    """
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM

def numpy_nms(boxes :array, scores :array, iou_threshold :float):

    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)

        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]

    keep = np.array(keep)  
    return keep
 
box =  np.array([[2,3.1,7,5],[3,4,8,4.8],[4,4,5.6,7],[0.1,0,8,1]]) 
score = np.array([0.5, 0.3, 0.2, 0.4])
 
output = numpy_nms(boxes=box, scores=score, iou_threshold=0.3)
print(output)

```

##  3.pytorch 实现线性回归

```python
import torch
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn
 
 
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 3*x + 10 + torch.rand(x.size())
# 上面这行代码是制造出接近y=3x+10的数据集，后面加上torch.rand()函数制造噪音
# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1) # 输入和输出的维度都是1
    def forward(self, x):
        out = self.linear(x)
        return out
 
if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()
 
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
num_epochs = 100

for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x).cuda()
        target = Variable(y).cuda()
    else:
        inputs = Variable(x)
        target = Variable(y)
 
    # 向前传播
    out = model(inputs)
    loss = criterion(out, target) 
    # 向后传播
    optimizer.zero_grad() # 注意每次迭代都需要清零
    loss.backward()
    optimizer.step()
 
    if (epoch+1) %20 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1, num_epochs, loss.data[0]))        
model.eval()
if torch.cuda.is_available():
    predict = model(Variable(x).cuda())
    predict = predict.data.cpu().numpy()
else:
    predict = model(Variable(x))
    predict = predict.data.numpy()
plt.plot(x.numpy(), y.numpy(), 'ro', label='Original Data')
plt.plot(x.numpy(), predict, label='Fitting Line')
plt.show()
```

# c++

## 1.求浮点数的算术平方根

```c++
#include <iostream>
#include <math.h>
using namespace std;
int main()
{
   double n;
   cin>>n;
   double l=0,r=n;
   double mid=(l+r)/2;
   while(fabs(mid*mid-n)>=1e-6){
       if(mid*mid> n) r=mid;
       else l=mid;
       mid=(l+r)/2;
   }
	cout<<mid<<endl;
    // printf("%.5lf",mid);
    return 0;
}
```

## 2.给定一个数n，m个完全平方数相加成n，求最小m。（ leetcode279）

```c++
#include <iostream>
#include <vector>
using namespace std;
int main()
{
   int n=0;
   cin>>n;
   vector<int> dp(n+1,0);
   for(int i=1;i<=n;i++){
       dp[i]=i;
       for(int j=1;j*j<=i;j++){
           dp[i]=min(dp[i],dp[i-j*j]+1);
       }
   }
   cout<<dp[n]<<endl;
   return 0;
}

```

## 3.共享指针数据结构

```c++
#include <iostream>
#include <vector>
using namespace std;
#include <iostream>
#include <memory>
using namespace std;
template <typename T>
class SharedPtr{
private:
    T *_ptr;
    int *_pcount;
public:
    SharedPtr(T *ptr= nullptr):_ptr(ptr),_pcount(new int(1)){}//构造函数
    SharedPtr(const SharedPtr &s):_ptr(s._ptr),_pcount(s._pcount){//用另一个共享指针初始化共享指针
        (*_pcount)++;
    }
    SharedPtr<T> &operator=(const SharedPtr &s){//重载=
        if(this != &s)
        {
            if(--(*(this->_pcount))==0)//
            {
                delete this->_ptr;
                delete this->_pcount;
            }
            _ptr=s._ptr;
            _pcount=s._pcount;
            *(_pcount)++;
        }
        return *this;
    }
    T &operator*()//重载*
    {
        return *(this->_ptr);
    }
    T *operator->()//重载->
    {
        return this->_ptr;
    }
    ~SharedPtr()//析构函数
    {
        --(*(this->_pcount));
        if(*(this->_pcount)==0)
        {
            delete _ptr;
            _ptr= nullptr;
            delete _pcount;
            _pcount= nullptr;
        }

    }

};
int main() {
    std::shared_ptr<int> p1(new int(4));
    cout<<"p1: "<<p1<<" *p1 "<<*p1;
    cout<<endl;

    return 0;
}

```

## 4.二叉树的最大路径和

```c++
struct TreeNode{
	struct TreeNode *left;
    struct TreeNode *right;
    int val;
    TreeNode(int x):val(x),left(nullptr),right(nullptr){}
    
};
int maxPathSum(TreeNode* root) {
        if(!root)return 0;
        int m=INT_MIN;  //存放最大路径和
        
        helper(root,m); //将m传入，便于更新最大值m
        
        return m;
    }
    
    int helper(TreeNode* root,int &m) //计算过当前结点的最大路径和
    {
        if(!root)return 0;
        
        int l=helper(root->left,m);   //过当前结点左子结点的最大路径和
        int r=helper(root->right,m);  //过当前结点右子结点的最大路径和
        
        int curSum=max(root->val,max(l+root->val,r+root->val));  //过当前结点的最大路径和
 
        int curMax=max(curSum,l+r+root->val); //如果将当前结点作为根结点，就要考虑横跨的情况
            
        m=max(m,curMax); //更新最大值
        
        return curSum; //返回过当前结点的最大路径和
    }

```

