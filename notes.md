
**<font face="宋体" size=7>$$机器学习算法总结$$</font>**
#1 引言
&emsp;　机器学习算法种类繁多，主要分为监督学习，非监督学习，强化学习。其中现在比较火热的深度学习又单独列出作为一个独立的方向。监督学习是指已知样本类别，通过算法学习模型。非监督学习是样本类别未知，算法自己学习。强化学习是介于监督学习和非监督学习之间，在学习过程中通过奖励信号来反馈是否做得对。本文主要总结机器学习中的监督学习和非监督学习方法，强化学习和深度学习暂不涉及。
#2 监督学习(Supervised Learning)
&emsp;　监督学习算法主要分为判别分析算法和生成学习算法。这两类算法是通过样本数据分别学习判别模型和生成模型。
&emsp;　判别学习算法是运用数据来学习条件概率分布$P(y|x)$或者直接学习目标函数$h_\theta(x)$算法直接考虑$P(y|x)$的极大似然估计从而求得模型$h_\theta(x)$的参数。在预测阶段，直接用习得的模型进行预测，即利用学习好的目标函数$h_\theta(x)$来进行预测。
&emsp;　生成学习算法运用数据来学习联合概率分布$P(x,y)$，得到各个类别的概率分布模型。算法考虑的是$P(x,y)=P(x|y)P(y)$极大似然估计，从而得到每个类别数据模型$P(x│y)$的参数。在预测阶段，算法是运用贝叶斯公式计算$P(y│x)=\frac{P(x|y)P(y)}{P(x)}$看哪个类别的$P(y│x)$值大即属于哪个类别。所以只能解决分类问题，不能用于回归问题。
##2.1 判别学习算法(Discriminative Learning Algorithms)
&emsp;　判别学习算法通过求解$P(y│x)$的极大似然估计来得到模型$h_\theta(x)$的参数，从而直接得到分类模型，即直接求解最优分类面。
&emsp;　**数学原理**：在统计中，大部分概率分布(Gaussian, Multigaussian, Bernoulli, Multinomial, Poisson)都可以写成指数分布族(Exponential Family)的形式，为$P(y,\eta)=b(y)exp⁡(\eta^T T(y)-a(\eta))$。所以判别学习中的概率$P(y│x)$在满足不同分布时也可以写成指数分布族的形式。将具体的分布同时取对数和指数，然后进行变换，即可求出$a,b,T$。这样通过指数分布族可以推导出广义线性模型(GLM, Generalized Linear Model)，满足三条假设。
&emsp;　当$P(y│x)$满足不同分布时，可以推导出不同的$h_\theta(x)$的形式，对应着不同的算法，例如：
&emsp;　满足Bernoulli 概率分布时，此时的$y$为二值(0,1)，可推导出$h_\theta(x)=\frac{1}{1+e^{(-\theta^Tx)}}$即为逻辑回归(Logistic Regression)。 
&emsp;　满足Gaussian概率分布时，此时的$y$为连续变量，$h_\theta(x)=\theta^Tx$为线性回归(Linear Regression)，多项式回归(Polynomial Regression)也是线性回归，为线性组合。
&emsp;　满足Multinomial概率分布时，此时的$y$为多类，$k$个值，$h_\theta(x)$形式见笔记，为softmax回归。
&emsp;　梯度下降法(或牛顿法)：求解$P(y│x)$的极大似然估计来得到模型$h_\theta(x)$的参数时，在概率统计方法里会用导数等于0的方法求极值。在计算机算法中会用到梯度下降法或牛顿法。
&emsp;　可以证明指数族分布的极大似然函数是concave的，所以有最大值。(关于concave和convex的定义见笔记P11)
###2.1.1 线性回归(Linear Regression)
&emsp;　将样本拟合成连续函数，之前论证过线性回归的目标函数为$h_\theta(x)=\theta^T x+\epsilon$。当目标函数的误差项$ε^{(i)}\sim N(0,\sigma^2)$满足正态分布时，对$P(y│x)$的极大似然估计求解可以推导(推导见笔记)出是对$J(\theta)=\frac{\sum^{m}_{i=1}{(y^i-\theta^T x)^2 }}{2}$求极小，即为最小二乘法(可用梯度下降法求解)。也可以用矩阵计算来求解$J(\theta)$的最小值。
####&emsp;　加权线性回归(LWR)
&emsp;　将每个样本加一个权值(一般权值满足正态分布)，在预测一个未知点$x$时，将$x$代入权值表达式中，得到权值(一般权值范围为从0到1)。然后对加权$J(\theta)$最小化求出系数$\theta^T$，从而得到由未知点周围样本生成的直线。通过此直线来预测未知样本的值。此方法是在预测时动态生成模型。
###2.1.2 逻辑回归(Logisitc Regression)
&emsp;　逻辑回归是用于分类，可以通过指数族推导出$h_\theta(x)=\frac{1}{1+e^{-\theta^T x}}$即在线性回归目标函数上进行函数变换，此变换叫sigma function。具体方法同线性回归，对$P(y│x)$的极大似然估计求解，但是$y$即$h_\theta (x)$满足Bernoulli 概率分布。推导出极大似然函数，然后用梯度下降法求最大值，即$\theta=\theta+\alpha(y-h_\theta(x))x_j$ (与线性回归思路一样，不过线性回归是求$J(\theta)$的最小值)。预测时，因为sigma 函数值是在0和1之间的，所以用0.5为分界。
###2.1.3 感知器学习算法(Perceptron Learning Algorithm)
&emsp;　与逻辑回归类似，用阶梯函数对线性回归的目标函数进行变换，使得新的目标函数$h_\theta(x)=g(\theta^T x)$输出0和1。同样，求极大似然估计，用梯度下降法求最大值。
###2.1.4 多项式分布回归(Softmax Regression)
&emsp;　这里要注意多项式回归与多项式分布回归的区别：多项式回归是在线性回归的基础上，对每一维变量扩展成幂的形式，这样解决了线性不可分的情况。而多项式分布回归是解决的多类分类的问题，即$y$满足Multinomial概率分布。(推导得出参数形式见笔记)
###2.1.5 求极值的算法
&emsp;　在数学中，求极值一般是求一次导数并领其为$\theta$，然后求解未知数。具体到计算机中实现，一般是用梯度下降法或者牛顿法。
####&emsp;　梯度下降法(Gradient Descent)：
&emsp;　选好$\theta$的初始值，然后通过$\theta_j≔\theta_j-\alpha\frac{\partial }{\partial \theta_j} J(\theta)$迭代，其中$theta_j$是第$j$个参数。通过计算可以得到$\theta_j≔\theta_j-\alpha(h_\theta(x)-y)x_j$在迭代部分，有两种方法，一种是批梯度下降法(Batch Gradient Descent)，$\theta_j≔\theta_j-\alpha \sum^{m}_{i=1}{(h_\theta(x^i)-y^i)x_j^i}$计算所有样本后再更新$\theta_j$ 另一种是随机梯度下降(Stochastic Gradient Descent)，一个样本把所有$\theta_j$都更新一边，然后再计算下一个样本。此方法比第一种收敛快，尤其是大样本时比批梯度下降快。梯度下降法依赖于初值，所以容易进入局部最小值。(但是如果本身函数是凸函数，即只有一个最值，那么算法是可以达到全局最优的。)
&emsp;　**数学原理**：由泰勒展开$f(\theta)=f(\theta_0)+(\theta-\theta_0)f^{\prime}(\theta_0)$,可得$\theta-\theta_0=\eta v$(其中，$\eta$标量，$v$单位矢量) 所以$f(\theta)=f(\theta_0)+\eta v f^{\prime}(\theta_0)$。若要求$f$的最小值，即$f(\theta)< f(\theta_0)$ 即$f(\theta)-f(\theta_0)=\eta v f^{\prime}(\theta_0)<0$ 因为$\eta$是标量，所以$v f^{\prime}(\theta_0)<0$ 根据两个向量相乘的公式得到，$vf^{\prime}(\theta_0)=\|v\| \|f^{\prime}(\theta_0)\|cos\alpha$要使$f(\theta)$最小，即$f(\theta)-f(\theta_0)$最大程度的小，所以$cos\alpha=-1$时才可。所以$v$与$f^{\prime}(\theta_0)$方向相反，即$v=-\frac{f^{\prime}(\theta_0)}{\|f^{\prime}(\theta_0)\|}$ 所以$\theta-\theta_0=\eta v$即$\theta=\theta_0+\eta v=\theta_0-\eta\frac{f^{\prime}(\theta_0)}{\|f^{\prime}(\theta_0)\|}$而$\frac{\eta}{\|f^{\prime}(\theta_0)\|}$为一个标量，即上式可以写为$\theta=\theta_0-\eta f^{\prime}(\theta_0)$以上推导只是一维变量，多维类似。
####&emsp;　牛顿法(Newton's Method)：
&emsp;　首先，对于一个函数$f$，如何求$\theta$使$f(\theta)=0$即$\frac{f(\theta)-f(\theta_0)}{(\theta-\theta_0 )}=f^{\prime}(\theta_0)$，因为$f(\theta)=0$，所以上式为$\theta=\theta_0-(f(\theta_0))/(f(\theta_0))$若对极大似然函数求极值，即极大似然函数导数为0，那么$f(\theta)=l^{\prime}(\theta)$，可带入上式。于是可求出θ为何值时极大似然函数值最大。牛顿法优点是迭代速度快，但当θ为向量时，求$l^{\prime\prime}(\theta)$需要计算海森矩阵。
###2.1.6 支持向量机(Support Vector Machines)
&emsp;　支持向量机被认为是最好的监督学习算法，它可以通过凸优化来达到全局最优，避免神经网络中的容易陷入局部最优的问题。
&emsp;　**第一种情况：线性可分**：记$y\in\{-1,+1\}$，$g(z)=1$当$z\geq0$时，$g(z)=-1$当$z$为其他。则目标函数为$h_{w,b}(x)=g(w^T x+b)$，实际上$w^T x+b$相当于线性回归里的$\theta^Tx$，因为$b$项相当于$\theta_0$。此算法的目的是当线性可分时，在所有样本正确分类的情况下，样本到分类超平面的几何间隔最大。
&emsp;　**函数间隔与几何间隔**：每一个样本关于超平面的函数间隔为$\hat{\gamma}=y^{(i)} (w^T x^{(i)}+b)$。为了获得最大函数间隔，若$y^(i)=1$则需要$w^T x^{(i)}+b\gg0$，若$y^(i)=-1$则需要$w^T x^{(i)}+b\ll0$，若样本的函数间隔$y^{(i)} (w^T x^{(i)}+b)>0$则说明样本正确分类了。整个样本集的函数间隔为最小的样本函数间隔，$\hat{\gamma}=min\hat{\gamma}$。这里函数间隔的意义是保证样本的正确分类。因为$g(w^T x+b)=g(2w^T x+2b)$所以加上一个约束$\|w\|=1$。即${\hat\gamma^{(i)}=y^{(i)}(\frac{w}{\|w\|}}^Tx^{(i)}+\frac{b}{\|w\|})$
&emsp;　几何间隔可以度量训练样本与超平面的几何距离。因为$w^Tx+b=0$平面的法向量为$w^T$写成列即为$w$，所以单位法向量为$\frac{w}{\|w\|}$，那么如图所示，$\overrightarrow{B}=\overrightarrow{A}-\gamma^{(i)}\frac{w}{\|w\|}=x^{(i)}-\gamma^{(i)}\frac{w}{\|w\|}$因为$B$在平面上，所以$w^T(x^{(i)}-\gamma^{(i)}\frac{w}{\|w\|})+b=0$，推导得$\gamma^{(i)}=\frac{w^Tx^{(i)}+b}{\|w\|}={\frac{w}{\|w\|}}^Tx^{(i)}+\frac{b}{\|w\|}$以上推导为样本为正类时，即$y=1$，若样本为负类时$\gamma$方向相反。所以几何间隔可以定义为${\gamma^{(i)}=y^{(i)}(\frac{w}{\|w\|}}^Tx^{(i)}+\frac{b}{\|w\|})$。因此当约束$\|w\|=1$时，函数间隔和几何间隔相同。同样，整个样本集的几何间隔为所有样本中最小的几何间隔值。
![Alt text](\01.jpg)![Alt text](\02.jpg)
&emsp;　**最大间隔分类器**
&emsp;　$max_{\gamma,w,b}\quad\gamma$
&emsp;　s.t. $\quad y^{(i)}(w^T x^{(i)}+b)\ge\gamma, i=1,...,m$
&emsp;　$\qquad \|w\|=1$
&emsp;　目标是求解$w,b$，每个样本都满足上式时(所有样本都能正确分类)，使得$\gamma$最大。约束$\|w\|=1$就是为了让几何间隔与函数间隔相同，即$\frac{\hat\gamma}{\|w\|}=\gamma$所以如上表达可以写为：
&emsp;　$max_{\gamma,w,b}\quad\frac{\hat\gamma}{\|w\|}$
&emsp;　s.t. $\quad y^{(i)}(w^T x^{(i)}+b)\ge\hat\gamma, i=1,...,m$
&emsp;　这里因为超平面为$w^T x+b=0$所以同时缩放$w,b$不影响超平面的位置，超平面位置不变，所以几何间隔的值也不变，所以上式还可以用$w,b$表示。此时，可以加约束$\hat\gamma=1$(如右图) 因为如果$\hat\gamma$为1的$m$倍，则$w,b$除以$m$，缩放后位置不变，同时$m$倍的最大值，也是原数的最大值。所以上式可以表达为：
&emsp;　$max_{\gamma,w,b}\quad\frac{1}{\|w\|}$
&emsp;　s.t. $\quad y^{(i)}(w^T x^{(i)}+b)\ge1, i=1,...,m$
&emsp;  道理同上，也可以继续缩放，将上式转化为$\frac{2}{\|w\|}$的最大值，即为$\frac{\|w\|}{2}$的最小值，因为$\|w\|>0$，所以$\frac{\|w\|}{2}$的最小值也是$\frac{\|w\|^2}{2}$(结合二次函数的图形)。所以上述表达可以写为：
&emsp;  $min_{\gamma,w,b}\quad\frac{1}{2} \|w\|^2$
&emsp;　s.t. $\quad y^{(i)}(w^T x^{(i)}+b)\ge1, i=1,...,m$
&emsp;　从而得到了如上的最优间隔分类器。
&emsp;　**Lagrange乘数法**
&emsp;　在求解函数的最值时一般都用到Largrange乘数法方法。最优化问题有三种形式: 无约束，只有等式约束，等式约束和不等式约束都有。
&emsp;　第一种，无约束，形式为$min_w f(w)$直接求偏导解方程得出$w$，代入函数中看是否是最值即可。
&emsp;　第二种，只有等式约束，形式为
&emsp;　$min_w f(w)$
&emsp;　s.t. $h_i(w)=0, i=1,...,l$
&emsp;　在求解时写成Largrange函数$L(w,\beta)=f(w)+\sum^l_{i=1}\beta_ih_i(w)$然后求偏导令$\frac{\partial L}{\partial {w_i}}=0,\frac{\partial L}{\partial {\beta_i}}=0$，解出$w,\beta$代入函数中，得到最值。
&emsp;　第三种，等式约束和不等式约束都有
&emsp;　$min_w f(w)$
&emsp;　s.t. $g_i(w)\leq 0$
&emsp;　&emsp; $h_i(w)=0, i=1,...,l$
&emsp;　写成广义Largrange函数$L(w,\alpha,\beta)=f(w)+\sum^k_{i=1}\alpha_ig_i(w)+\sum^l_{i=1}\beta_ih_i(w)$解此类问题时，需要满足KKT条件，然后将原始问题写成对偶问题。
&emsp;　**原始问题和对偶问题**
&emsp;　如上所述的第三种情况的形式叫做原始问题。由于第三种情况的约束比较复杂，所以要想办法把约束去掉。于是对广义Largrange函数$L(x,\alpha,\beta)=f(x)+\sum^k_{i=1}\alpha_ig_i(x)+\sum^l_{i=1}\beta_ih_i(x)$求最大值，对$\alpha,\beta$求最大值，并约定$\alpha_i>0$。在这里$\alpha,\beta$为变量，$x$为常量。所以最大值记为$\theta_p(x)=max_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)$。下面考虑$x$是否满足约束条件，若不满足，即$g_i(x)>0$或$h_i(x)\neq0$，则$\theta_p(x)=max_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)=+\infty$。因为当$g_i(x)>0$且$\alpha_i>0$，则函数$L$的第二项为无穷大。若$x$满足约束条件，则$L$函数的第三项为0，第二项最大值也为0，而第一项$f(x)$是常量。所以$\theta_p(x)=max_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)=f(x)$
&emsp;　总结一下，像上一节第三种形式的原始问题，可以写为$min_x f(x)=min_xmax_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)=min_x\theta_p(x)$可以记为$p^*$，即$p^*=min_x\theta_p(x)$为原始问题的最优值。
&emsp;　下面再看一下对偶问题，求上面Largrange函数关于$x$的最小值，即$\theta_D(\alpha,\beta)=min_xL(x,\alpha,\beta)$，这是关于$\alpha,\beta$的函数。考虑最大值$max_{\alpha,\beta:\alpha_i\ge0}\theta_D(\alpha,\beta)=max_{\alpha,\beta:\alpha_i\ge0}min_xL(x,\alpha,\beta)$这就是原始问题的对偶问题。形式上与原始问题是对称的。原始问题是先优化$\alpha,\beta$再优化$x$，对偶问题是先优化$x$，再优化$\alpha,\beta$。对偶问题的最优值记为$d^*=max_{\alpha,\beta}\theta_D(\alpha,\beta)$
&emsp;　下面看原始问题与对偶问题的关系。有定理，原始问题的最优值不小于对偶问题的最优值，即$d^*\leq p*$。证明：对任意$\alpha,\beta,x$，有$\theta_D(\alpha,\beta)=min_xL(x,\alpha,\beta)\leq L(x,\alpha,\beta)\leq max_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)=\theta_p(x)$即$\theta_D(\alpha,\beta)\leq \theta_p(x)$，再对对偶问题和原始问题取最优值，即分别取最大值和最小值，得到$max_{\alpha,\beta:\alpha_i\ge0}\theta_D(\alpha,\beta)\leq min_x\theta_p(x)$，即$d^*\leq p^*$。证明完毕。那么如果等号成立，即原始问题的最优值和对偶问题的最优值相等时，就可以通过求解对偶问题的最优值来得到原始问题的最优值。满足什么样的条件等号成立呢，就是下面要说的KKT条件。
&emsp;　**KKT条件**
&emsp;　首先，如果等号成立即$d*=p*$，即原始问题的最优解和对偶问题的最优解相等，假设最优解为$(x^*,\alpha^*,\beta^*)$。那么此解必然满足原始问题的约束和对偶问题的约束。即$g_i(x^*)\leq0$，$h_i(x^*)=0$，$\alpha_i^*\ge0$,同时，由于存在最优解，所以对变量的偏导数为0，即$\nabla_xL(x^*,\alpha^*,\beta^*)=0，\nabla_{\alpha}L(x^*,\alpha^*,\beta^*)=0，\nabla_{\beta}L(x^*,\alpha^*,\beta^*)=0$。同时，还能得到最重要的一条$\alpha_i^*g_i(x^*)=0$。以上就是KKT条件。下面说明为什么$\alpha_i^*g_i(x^*)=0$。因为$\alpha_i^*\ge0$，若$\alpha_i^*>0$则$g_i(x^*)=0$。因为$g_i(x^*)\leq0$，若$g_i(x^*)<0$，则$\alpha_i^*=0$。就是说$\alpha_i^*$和$g_i(x^*)$必然有一个为0，所以乘积为0。
&emsp;　**线性可分支持向量机的推导**
&emsp;  在之前已经提到了，线性可分支持向量机的问题就是求解如下最优化问题
&emsp;  $min_{\gamma,w,b}\quad\frac{1}{2} \|w\|^2$
&emsp;　s.t. $\quad y^{(i)}(w^T x^{(i)}+b)\ge1, i=1,...,m$
&emsp;  因为上述约束中只有不等式形式，所以Lagrange乘子只有$\alpha_i$，$g_i(w,b)=-y^{(i)}(w^Tx^(i)+b)+1$，由KKT条件可以得到当$\alpha_i>0$时，$g_i(w,b)=0$，即此样本距离分类平面的函数间隔为1。此类样本距离分类平面最近，叫做支持向量。而$\alpha_i=0$的样本是非支持向量。实际上，非支持向量占大多数。参考图2。下面写出Lagrange函数，$L(w,b,\alpha)=\frac{1}{2}\|w\|^2-\sum^m_{i=1}\alpha_i(y^{(i)}(w^Tx^{(i)}+b)-1)$。整个思路是求解出对偶问题的最优解，根据定义，对偶问题为$\theta_D(\alpha)=min_{w,b}L(w,b,\alpha)$即求Lagrange函数的极值点，然后代入，便得到$\theta_D(\alpha)$。然后再求最大值即为对偶函数的最优值。$\nabla_wL(w,b,\alpha)=w-\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)}$令其为0，解得$w=\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)}$同理，$\nabla_bL(w,b,\alpha)=-\sum^m_{i=1}y^{(i)}\alpha_i$令其为0得$\sum^m_{i=1}y^{(i)}\alpha_i=0$然后将$w$带入Lagrange函数得$L(w,b,\alpha)=\frac{1}{2}(\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)})^T(\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)})-\sum^m_{i=1}\alpha_i(y^{(i)}(w^Tx^{(i)}+b)-1)$因为$\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)}$为常数，所以转置等于其本身，即$L(w,b,\alpha)=\frac{1}{2}(\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)})(\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)})-\sum^m_{i=1}\alpha_i(y^{(i)}(w^Tx^{(i)}+b)-1)=\frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}y^{(i)}y^{(j)}\alpha_i\alpha_j<x^{{i}},x^{(j)}>-\sum^m_{i=1}\sum^m_{j=1}y^{(i)}y^{(j)}\alpha_i\alpha_j<x^{{i}},x^{(j)}>+\sum^m_{i=1}\alpha_i$第二项是由$-\sum^m_{i=1}\alpha_i(y^{(i)}(w^Tx^{(i)}+b)-1)=-\sum^m_{i=1}\alpha_iy^{(i)}w^Tx^{(i)}-b\sum^m_{i=1}\alpha_iy^{(i)}+\sum^m_{i=1}\alpha_i$因为$\sum^m_{i=1}\alpha_iy^{(i)}=0$同时将$w$的表达式代入即为所得。于是$L(w,b,\alpha)=\sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}y^{(i)}y^{(j)}\alpha_i\alpha_j<x^{{i}},x^{(j)}>$令其为$W(\alpha)$，这其实就是$L(w,b,\alpha)$关于$w,b$的最小值。
&emsp;　下面再考虑$W(\alpha)$关于$\alpha$的最大值


&emsp;　**kernels**

&emsp;　**第二种情况，线性不可分**

&emsp;　几个问题：SVM中的过拟合，小样本问题




##2.2 生成学习算法(Generative Learning Algorithms)
&emsp;　生成学习算法是对各个类分别建模，通过学习，建立每个类别的$P(x|y)$的概率模型(通过极大似然估计求解模型参数),同时通过样本计算每个类别的$P(y)$，从而得到$P(x,y)=P(x|y)P(y)$。在预测时，利用贝叶斯公式，$P(y=i|x)=\frac{P(x|y=i)P(y)}{P(x)}$,其中$p(x)=\sum_i{P(x|y=i)P(y=i)}$，求得样本$x$对每一类$y=i$的概率，从而得到$x$属于哪一类。
###2.2.1 高斯判别分析(Gaussian discriminant analysis)
&emsp;　高斯判别分析是生成学习算法的一种，是假设每一类$P(x|y=i)$都符合高斯分布。将每一类的密度函数写出，在两类时，参数为$\phi,\mu_0,\mu_1,\Sigma$，利用极大似然估计得出参数，从而得出每个类别的高斯模型。在预测时，由于$P(x)$值与将$x$分为哪一类无关，所以分类时比较$P(y=i|x)$大小，相当于比较$P(x|y=i)P(y)$大小，同时若各类的$y$是均匀分布，则相当于比较各类的$P(x|y=i)$大小。
&emsp;　高斯判别分析与逻辑回归的关系:如果$x|y$满足高斯分布(或者其他分布，如泊松分布)，且$y$为二值两类，则肯定满足逻辑回归。反之不成立。因此，若能肯定模型满足高斯分布，则高斯判别分析要比逻辑回归效果好，若不能肯定，则逻辑回归效果好。
###2.2.2 朴素贝叶斯(Naive Bayes)
&emsp;　此方法的算法思路与其他生成学习算法一样(见2.2)。假设样本之间各维是独立同分布的。同样，由于在预测时与$P(x)$无关，所以只考虑$P(x|y=i)P(y)$
&emsp;　根据样本$x$的每一维满足的概率分布，朴素贝叶斯又有三种模型：伯努利模型，多项式模型，高斯模型。
&emsp;　伯努利模型中，样本的每一维$P(x=j|y=i)$满足伯努利分布(即只要0，1二值)，同时由于各个维是独立同分布的，那么对相当于$P(x_j,y_i)=P(x=j|y=i)P(y=i)=\prod^n _j P(x_j)P(y=i)$的参数进行极大似然估计。在伯努利模型中估计参数$\phi_{j|y=i}$(即每一个$y$的类别中，$x$的各个维出现的概率)以及参数$\phi_{y=i}$(即每一类的概率)
&emsp;　多项式模型，思路和伯努利模型类似。在这里，样本的每一维$P(x=j|y=i)$满足多项式分布(取$k$个值)。
&emsp;　高斯模型，样本的每一维$P(x=j|y=i)$满足高斯分布，即每一维都是连续的，同样进行极大似然估计求出参数(这里注意与高斯判别分析相区别)。另一种处理方法是将连续值离散化。
#3 非监督学习(Unsupervised Learning)
Clustering K-mean(为什么线性回归可以达到全局最优，而kmean不行？)
Mixture of Gaussians, EM
Factor analysis
PCA
ICA
#4 Learning Theory学习理论适用的范围，是所有模型都适用？
Bias/variance tradeoff
Model selection and feature selection
#5 基本概念

##5.1 监督学习与非监督学习

##5.2 判别学习与生成学习

##5.3 线性模型与非线性模型 （分类具体模型是线性还是非线性）
&emsp;　线性与否不是指目标函数$h_\theta(x)$是否是线性的而是$h_\theta(x)$的参数是否是线性的。即观察x中的每一维看是不是只被一个参数影响，如果是即为线性，如果不是就是非线性。
&emsp;　如，$y=\frac{1}{1+e^{(w_0+w_1*x_1+w_2*x_2 )}}$是线性的，$y=\frac{1}{1+w_5*e^{(w_0+w_1*x_1+w_2*x_2 )}}$是非线性的
##5.4 参数学习与非参数学习
&emsp;　参数学习是模型通过训练求得参数，此参数固定不变。而非参数学习是动态求得参数。
&emsp;　参数学习：
&emsp;　非参数学习：加权线性回归
##5.5 概率模型与非概率模型

##5.6 回归与分类
&emsp;　回归模型离散化后即可得到分类模型。
#6 机器学习算法的评价

$\frac{\partial x}{\partial y}$
$\alpha$
$\sum^{m}_{i=1}{\frac{x}{y}}$
$\displaystyle \sum^{x \to \infty}_{y \to 0}{\frac{x}{y}}$
$\ddot{a}$
$\sim$