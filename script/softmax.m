% 定义一个归一化向量
x = [1, 2, 3, 9, 10];

% 计算 softmax
ex = exp(x);
softmax_x = ex./sum(ex);


plot(x); 
hold on;
plot(softmax_x);
legend('x','softmax');

ax = gca;
ax.FontName = "Times New Roman";