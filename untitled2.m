close all
p=importdata("20-(2e4)_seed5.txt");

ti=400;

aa=p(:,2);
bb=p(:,3);
for i=1:ti
    aa=smooth(aa);
    bb=smooth(bb);
end

figure
%plot(p(:,1),p(:,2),'r')
plot(p(:,1),p(:,2), 'r','LineWidth', 0.5);hold on
plot(p(:,1),aa,'r','LineWidth', 2,'DisplayName','PPO');hold on
title("奖励曲线")
h = gca;
h.Children(1).Color(4) = 1;
h.Children(2).Color(4) = 0.1;
hold on
figure
%plot(p(:,1),p(:,3),'g')
plot(p(:,1),p(:,3), 'b','LineWidth', 0.5);hold on
plot(p(:,1),bb,'b','LineWidth', 2,'DisplayName','PPO(不归一化)');hold on
title("准确率曲线")

h = gca;
h.Children(1).Color(4) = 1;
h.Children(2).Color(4) = 0.1;
hold on

