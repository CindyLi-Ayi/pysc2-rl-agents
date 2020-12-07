# pysc2-rl-agents

Homework2 For Reinforcement Learning（本次实验平均分都是在100轮取mean）

补充说明：关于几种rulr-base方法的实现以及最终得分：

# FindAndDefeatZerglings 
best_mean_score = 25(rule based) 

选择出地图上的一些基本点（走完这些基本点构成的路径会遍历地图的所有盲区），选择全体marine按照这些基本点的顺序进行queued_attack_minimap保证将地图遍历，同时如果在视野中发现zerglings就优先进行attack，但是这种方法的缺点是容易一次性攻击多个敌人导致全部死亡，更好的策略是尽可能保证一次只攻击一个敌人。

# DefeatRoaches
best_mean_score =  171（out rule-base model）

Rule base的想法是在开局阶段（根据多出4个兵判断为开局），计算自己和敌人的平均相对位置，如果在敌人偏上方，则先向上进行一次移动，然后按照从上到下的顺序集火攻击敌人，反之若在偏下方，则向下进行一次移动，然后从下到上集火攻击。这里进行一次移动的目的是调整自己的阵型位置，尽可能的减少暴露在对方射程范围内的兵。最后根据对比，这次位置调整的作用非常明显，不进行位置调整直接集火攻击平均105，而改进后的agent的平均得分为171.（只要前面死的少，每次获胜都得到4个兵，收益滚雪球式上升）

# DefeatZerglingsAndBanelings
best_mean_score = 213(our rule-baesd model)

Rule based的方法是利用对方会自爆的特性，每次开局（可判断获得4个兵也为开局）都选择最上和最下两个士兵去攻击对方的两只Baneling，使其自爆，其它marine静止不动避免被爆炸造成大规模损失，然后对剩下的敌人依次集火攻击，就可以巨大的优势胜利。（其中发现了一个小bug，由于这种策略优势巨大，最后获得的兵越来越多，超过一定数量就会挤出屏幕外，坐标为负，所以加入了特判避免越界）
