> https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management
> 耽误了很多时间才搞清楚现在的atari环境配置问题，能找到原始文档的还是看原始文档，百度、谷歌搜别人写的文档都太过时了

# 命令
首先把命令全部记下来：
```
conda create -n RL39 python=3.9
conda activate RL39
pip install gym
pip install gym[atari]
pip install gym[accept-rom-license]
```
# 出现的问题
从Gym v0.20开始，所有雅达利环境都是通过ale-py提供的（gym[atari]）。
此刻2022年2月，gym的版本是0.21.0，所以现在的语法和之前的有点不一样了。
这时候推荐在ALE名称空间中使用新的v5环境，新的环境可以随意改变游戏框的大小，也可能随意移动等等，按以下代码格式
```
import gym
env = gym.make('ALE/Breakout-v5')
```
经过几个环境的测试，就是在`ALE/`后面加上对应游戏的名字及版本`-v5`
如果不加`ALE/`直接用游戏名字其实也可以运行，但是会提示出游戏版本过时


# 区别
可以说唯一一个重要的区别就是新的版本不能之间使用`env.render()`
现在的版本如果想渲染出游戏画面，需要在创建环境的时候加上`render_mode`参数'human'，
还有一种模式是rgb_array，是在元数据字典中返回智能体行动之后返回的完整的RGB观察结果。
如下
```
import gym
env = gym.make('ALE/Breakout-v5', render_mode='human')
o = env.reset()
while True:
    a = env.action_space.sample()
    o, r, done, _ = env.step(a)
    print(o.shape)
    if done:
        break
env.close() # close and clean up
```
这样在训练的时候自动会渲染出游戏画面！

当然，想让游戏运行的时候不要渲染出画面也非常简单，就是`env = gym.make('ALE/Breakout-v5')`这样就好了
可以说，`env.render()`以后就不需要用了
