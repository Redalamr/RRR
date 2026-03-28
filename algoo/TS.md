

JMLR: Workshop and Conference Proceedings vol 23 (2012) 39.1–39.2625th Annual Conference on Learning Theory
Analysis of Thompson Sampling for the Multi-armed Bandit Problem
Shipra AgrawalSHIPRA@MICROSOFT.COM
## Microsoft Research India
Navin GoyalNAVINGO@MICROSOFT.COM
## Microsoft Research India
Editor:Shie Mannor, Nathan Srebro, Robert C. Williamson
## Abstract
The multi-armed bandit problem is a popular model for studying exploration/exploitation trade-off
in sequential decision problems. Many algorithms are now available for this well-studied problem.
One of the earliest algorithms,  given by W. R. Thompson,  dates back to 1933.  This algorithm,
referred to as Thompson Sampling, is a natural Bayesian algorithm.  The basic idea is to choose
an arm to play according to its probability of being the best arm.  Thompson Sampling algorithm
has experimentally been shown to be close to optimal. In addition, it is efficient to implement and
exhibits several desirable properties such as small regret for delayed feedback. However, theoretical
understanding of this algorithm was quite limited.  In this paper, for the first time, we show that
Thompson Sampling algorithm achieves logarithmic expected regret for the stochastic multi-armed
bandit problem.  More precisely, for the stochastic two-armed bandit problem, the expected regret
in timeTisO(
lnT
## ∆
## +
## 1
## ∆
## 3
). And, for the stochasticN-armed bandit problem, the expected regret in
timeTisO(
## [
## (
## ∑
## N
i=2
## 1
## ∆
## 2
i
## )
## 2
## ]
lnT).  Our bounds are optimal but for the dependence on∆
i
and the
constant factors in big-Oh.
Keywords:multi-armed bandit, Thompson Sampling, Bayesian algorithm, online learning
## 1.  Introduction
Multi-armed  bandit  problem  models  the  exploration/exploitation  trade-off  inherent  in  sequential
decision  problems.   Many  versions  and  generalizations  of  the  multi-armed  bandit  problem  have
been studied in the literature;  in this paper we will consider a basic and well-studied version of
this problem: the stochastic multi-armed bandit problem. Among many algorithms available for the
stochastic bandit problem, some popular ones include Upper Confidence Bound (UCB) family of
algorithms, (e.g., Lai and Robbins(1985);Auer et al.(2002), and more recentlyGarivier and Capp
## ́
e
(2011),Maillard et al.(2011),Kaufmann et al.(2012)), which have good theoretical guarantees, and
the algorithm byGittins(1989), which gives optimal strategy under Bayesian setting with known
priors and geometric time-discounted rewards.   In one of the earliest works on stochastic bandit
problems,Thompson(1933) proposed a natural randomized Bayesian algorithm to minimize regret.
The basic idea is to assume a simple prior distribution on the parameters of the reward distribution
of  every  arm,  and  at  any  time  step,  play  an  arm  according  to  its  posterior  probability  of  being
the best arm.   This algorithm is known asThompson Sampling(TS), and it is a member of the
family ofrandomized probability matchingalgorithms.  We emphasize that although TS algorithm
is a Bayesian approach,  the description of the algorithm and our analysis apply to the prior-free
stochastic multi-armed bandit model where parameters of the reward distribution of every arm are
c
## ©2012 S. Agrawal & N. Goyal.

## AGRAWALGOYAL
fixed, though unknown (see Section 1.1).  One could interpret the “assumed” Bayesian priors as
the current knowledge of the algorithm about the arms.   Thus,  our regret bounds for Thompson
Sampling are directly comparable to the regret bounds for UCB family of algorithms which are a
frequentist approach to the same problem.
Recently, TS has attracted considerable attention.  Several studies (e.g.,Granmo(2010);Scott
(2010);Chapelle and Li(2011);May and Leslie(2011)) have empirically demonstrated the efficacy
of Thompson Sampling:Scott(2010) provides a detailed discussion of probability matching tech-
niques in many general settings along with favorable empirical comparisons with other techniques.
Chapelle and Li(2011) demonstrate that empirically TS achieves regret comparable to the lower
bound ofLai and Robbins(1985); and in applications like display advertising and news article rec-
ommendation, it is competitive to or better than popular methods such as UCB. In their experiments,
TS is also more robust to delayed or batched feedback (delayed feedback means that the result of
a play of an arm may become available only after some time delay, but we are required to make
immediate decisions for which arm to play next) than the other methods.  A possible explanation
may be that TS is a randomized algorithm and so it is unlikely to get trapped in an early bad decision
during the delay.  Microsoft’s adPredictor (Graepel et al.(2010)) for CTR prediction of search ads
on Bing uses the idea of Thompson Sampling.
It has been suggested (Chapelle and Li(2011)) that despite being easy to implement and being
competitive to the state of the art methods, the reason TS is not very popular in literature could be
its lack of strong theoretical analysis.  Existing theoretical analyses inGranmo(2010);May et al.
(2011) provide weak guarantees, namely, a bound ofo(T)on expected regret in timeT.  In this
paper, for the first time, we provide a logarithmic bound on expected regret of TS algorithm in time
Tthat is close to the lower bound ofLai and Robbins(1985). Before stating our results, we describe
the MAB problem and the TS algorithm formally.
1.1.  The multi-armed bandit problem
We consider the stochastic multi-armed bandit (MAB) problem: We are given a slot machine with
Narms; at each time stept= 1,2,3,..., one of theNarms must be chosen to be played. Each arm
i, when played, yields a random real-valued reward according to some fixed (unknown) distribution
with support in[0,1].  The random reward obtained from playing an arm repeatedly are i.i.d.  and
independent of the plays of the other arms.  The reward is observed immediately after playing the
arm.
An algorithm for the MAB problem must decide which arm to play at each time stept, based
on the outcomes of the previoust−1plays. Letμ
i
denote the (unknown) expected reward for arm
i.  A popular goal is to maximize the expected total reward in timeT, i.e.,E[
## ∑
## T
t=1
μ
i(t)
], where
i(t)is the arm played in stept, and the expectation is over the random choices ofi(t)made by the
algorithm.  It is more convenient to work with the equivalent measure of expected totalregret:  the
amount we lose because of not playing optimal arm in each step.  To formally define regret, let us
introduce some notation. Letμ
## ∗
:= max
i
μ
i
, and∆
i
## :=μ
## ∗
## −μ
i
. Also, letk
i
(t)denote the number
of times armihas been played up to stept−1. Then the expected total regret in timeTis given by
## E[R(T)] =E
## [
## ∑
## T
t=1
## (μ
## ∗
## −μ
i(t)
## )
## ]
## =
## ∑
i
## ∆
i
·E[k
i
## (T)].
Other performance measures include PAC-style guarantees; we do not consider those measures here.
## 39.2

## ANALYSIS OFTHOMPSONSAMPLING
## 1.2.  Thompson Sampling
For simplicity of discussion, we first provide the details of Thompson Sampling algorithm for the
Bernoulli bandit problem, i.e.  when the rewards are either0or1, and for armithe probability of
success (reward =1) isμ
i
. This description of Thompson Sampling follows closely that ofChapelle
and Li(2011).  Next, we propose a simple new extension of this algorithm to general reward dis-
tributions with support[0,1], which will allow us to seamlessly extend our analysis for Bernoulli
bandits to general stochastic bandit problem.
The  algorithm  for  Bernoulli bandits  maintains  Bayesian  priors  on  the Bernoulli  meansμ
i
## ’s.
Beta distribution turns out to be a very convenient choice of priors for Bernoulli rewards.  Let us
briefly recall that beta distributions form a family of continuous probability distributions on the
interval(0,1).   The pdf of Beta(α,β),  the beta distribution with parametersα >0,β >0,  is
given byf(x;α,β) =
## Γ(α+β)
Γ(α)Γ(β)
x
α−1
## (1−x)
β−1
.  The mean of Beta(α,β)isα/(α+β); and as is
apparent from the pdf, higher theα,β, tighter is the concentration of Beta(α,β)around the mean.
Beta distribution is useful for Bernoulli rewards because if the prior is a Beta(α,β)distribution, then
after observing a Bernoulli trial, the posterior distribution is simply Beta(α+1,β)or Beta(α,β+1),
depending on whether the trial resulted in a success or failure, respectively.
The Thompson Sampling algorithm initially assumes armito have prior Beta(1,1)onμ
i
, which
is natural because Beta(1,1)is the uniform distribution on(0,1). At timet, having observedS
i
## (t)
successes (reward =1) andF
i
(t)failures (reward =0) ink
i
(t) =S
i
(t) +F
i
(t)plays of armi, the
algorithm updates the distribution onμ
i
as Beta(S
i
(t) + 1,F
i
(t) + 1). The algorithm then samples
from these posterior distributions of theμ
i
’s, and plays an arm according to the probability of its
mean being the largest. We summarize the Thompson Sampling algorithm below.
Algorithm 1Thompson Sampling for Bernoulli bandits
For each armi= 1,...,NsetS
i
## = 0,F
i
## = 0.
foreacht= 1,2,...,do
For each armi= 1,...,N, sampleθ
i
(t)from the Beta(S
i
## + 1,F
i
## + 1)distribution.
Play armi(t) := arg max
i
θ
i
(t)and observe rewardr
t
## .
Ifr= 1, thenS
i(t)
## =S
i(t)
+ 1, elseF
i(t)
## =F
i(t)
## + 1.
end
We adapt the Bernoulli Thompson sampling algorithm to the general stochastic bandits case,
i.e.  when the rewards for armiare generated from an arbitrary unknown distribution with support
[0,1]and meanμ
i
,  in a way that allows us to reuse our analysis of the Bernoulli case.   To our
knowledge, this adaptation is new.  We modify TS so that after observing the reward ̃r
t
## ∈[0,1]at
timet, it performs a Bernoulli trial with success probability ̃r
t
.  Let random variabler
t
denote the
outcome of this Bernoulli trial, and let{S
i
(t),F
i
(t)}denote the number of successes and failures
in the Bernoulli trials until timet.  The remaining algorithm is the same as for Bernoulli bandits.
Algorithm 2 gives the precise description of this algorithm.
We observe that the probability of observing a success (i.e.,r
t
= 1) in the Bernoulli trial after
playing an armiin the new generalized algorithm is equal to the mean rewardμ
i
## . Letf
i
denote the
(unknown) pdf of reward distribution for armi. Then, on playing armi,
## Pr(r
t
## = 1) =
## ∫
## 1
## 0
## ̃rf
i
## ( ̃r)d ̃r=μ
i
## .
## 39.3

## AGRAWALGOYAL
Algorithm 2Thompson Sampling for general stochastic bandits
For each armi= 1,...,NsetS
i
## (1) = 0,F
i
## (1) = 0.
foreacht= 1,2,...,do
For each armi= 1,...,N, sampleθ
i
(t)from the Beta(S
i
## + 1,F
i
## + 1)distribution.
Play armi(t) := arg max
i
θ
i
(t)and observe reward ̃r
t
## .
Perform a Bernoulli trial with success probability ̃r
t
and observe outputr
t
## .
## Ifr
t
= 1, thenS
i(t)
## =S
i(t)
+ 1, elseF
i(t)
## =F
i(t)
## + 1.
end
Thus, the probability of observingr
t
= 1is same andS
i
(t),F
i
(t)evolve exactly in the same way as
in the case of Bernoulli bandits with meanμ
i
. Therefore, the analysis of TS for Bernoulli setting is
applicable to this modified TS for the general setting.  This allows us to replace, for the purpose of
analysis, the problem with general stochastic bandits with Bernoulli bandits with the same means.
We  remark  that  instead  of  usingr
t
,  we  could  consider  more  direct  and  natural  updates  of  type
## Beta(α
i
## ,β
i
## )to Beta(α
i
## +  ̃r
t
## ,β
i
## + 1− ̃r
t
).  However, we do not know how to analyze this because
of our essential use of Fact1, which requiresα
i
## ,β
i
to be integral.
1.3.  Our results
In this article, we bound thefinite timeexpected regret of Thompson Sampling.  From now on we
will assume that the first arm is the unique optimal arm, i.e.,μ
## ∗
## =μ
## 1
>arg max
i6=1
μ
i
## . Assuming
that the first arm is an optimal arm is a matter of convenience for stating the results and for the
analysis.  The assumption ofuniqueoptimal arm is also without loss of generality, since adding
more arms withμ
i
## =μ
## ∗
can only decrease the expected regret; details of this argument are provided
in AppendixA.
Theorem 1For the two-armed stochastic bandit problem (N= 2), Thompson Sampling algorithm
has expected regret
## E[R(T)] =O
## (
lnT
## ∆
## +
## 1
## ∆
## 3
## )
in timeT, where∆ =μ
## 1
## −μ
## 2
## .
Theorem 2For theN-armed stochastic bandit problem, Thompson Sampling algorithm has ex-
pected regret
## E[R(T)]≤O
## 
## 
## (
## N
## ∑
a=2
## 1
## ∆
## 2
a
## )
## 2
lnT
## 
## 
in timeT, where∆
i
## =μ
## 1
## −μ
i
## .
Remark 3For theN-armed bandit problem, we can obtain an alternate bound of
## E[R(T)]≤O
## (
## ∆
max
## ∆
## 3
min
## (
## N
## ∑
a=2
## 1
## ∆
## 2
a
## )
lnT
## )
by slight modification to the proof. The above bound has a better dependence onNthan in Theorem
2, but worse dependence on∆
i
s. Here∆
min
= min
i6=1
## ∆
i
## ,∆
max
= max
i6=1
## ∆
i
## .
## 39.4

## ANALYSIS OFTHOMPSONSAMPLING
In interest of readability, we used big-Oh notation
## 1
to state our results.  The exact constants are
provided in the proofs of the above theorems.  Let us contrast our bounds with the previous work.
Lai and Robbins(1985) proved the following lower bound on regret of any bandit algorithm:
## E[R(T)]≥
## [
## N
## ∑
i=2
## ∆
i
## D(μ
i
## ||μ)
## +o(1)
## ]
lnT,
whereDdenotes  the  KL  divergence.   They  also  gave  algorithms  asymptotically  achieving  this
guarantee, though unfortunately their algorithms are not efficient.Auer et al.(2002) gave the UCB1
algorithm, which is efficient and achieves the following bound:
## E[R(T)]≤
## [
## 8
## N
## ∑
i=2
## 1
## ∆
i
## ]
lnT+ (1 +π
## 2
## /3)
## (
## N
## ∑
i=2
## ∆
i
## )
## .
For many settings of the parameters, the bound of Auer et al.  is not far from the lower bound of
Lai and Robbins.  Our bounds are optimal in terms of dependence onT, but inferior in terms of
the constant factors and dependence on∆.  We note that for the two-armed case our bound closely
matches the bound ofAuer et al.(2002). For theN-armed setting, the exponent of∆’s in our bound
is basically4compared to the exponent1for UCB1.
More recently,Kaufmann et al.(2012) gave Bayes-UCB algorithm which achieves the lower
bound ofLai and Robbins(1985) for Bernoulli rewards. Bayes-UCB is a UCB like algorithm, where
the upper confidence bounds are based on the quantiles of Beta posterior distributions. Interestingly,
these upper confidence bounds turn out to be similar to those used by algorithms inGarivier and
## Capp
## ́
e (2011) andMaillard et al.(2011).  Bayes-UCB can be seen as an hybrid of TS and UCB.
However, the general structure of the arguments used inKaufmann et al.(2012) is similar toAuer
et al.(2002); for the analysis of Thompson Sampling we need to deal with additional difficulties, as
discussed in the next section.
## 2.  Proof Techniques
In this section, we give an informal description of the techniques involved in our analysis. We hope
that this will aid in reading the proofs, though this section is not essential for the sequel. We assume
that all arms are Bernoulli arms, and that the first arm is the unique optimal arm.  As explained in
the previous sections, these assumptions are without loss of generality.
Main technical difficulties.Thompson Sampling is a randomized algorithm which achieves ex-
ploration by choosing to play the arm with best sampled mean, among those generated from beta
distributions around the respective empirical means. The beta distribution becomes more and more
concentrated around the empirical mean as the number of plays of an arm increases. This random-
ized setting is unlike the algorithms in UCB family, which achieve exploration by adding adeter-
ministic, non-negativebias inversely proportional to the number of plays, to the observed empirical
means. Analysis of TS poses difficulties that seem to require new ideas.
For example, following general line of reasoning is used to analyze regret of UCB like algo-
rithms  in  two-arms  setting  (for  example,  inAuer  et  al.(2002)):  once  the  second  arm  has  been
- For any two functionsf(n), g(n),f(n) =O(g(n))if there exist two constantsn
## 0
andcsuch that for alln≥n
## 0
## ,
f(n)≤cg(n).
## 39.5

## AGRAWALGOYAL
played sufficient number of times, its empirical mean is tightly concentrated around its actual mean.
If the first arm has been played sufficiently large number of times by then, it will have an empirical
mean close to its actual mean and larger than that of the second arm. Otherwise, if it has been played
small number of times, its non-negative bias term will be large. Consequently, once the second arm
has been played sufficient number of times, it will be played with very small probability (inverse
polynomial of time)regardless of the number of times the first arm has been played so far.
However, for Thompson Sampling, if the number of previous plays of the first arm is small, then
the probability of playing the second arm could be as large as a constant even if it has already been
played large number of times. For instance, if the first arm has not been played at all, thenθ
## 1
## (t)is
a uniform random variable, and thusθ
## 1
(t)< θ
## 2
(t)with probabilityθ
## 2
## (t)≈μ
## 2
. As a result, in our
analysis we need to carefully consider the distribution of the number of previous plays of the first
arm, in order to bound the probability of playing the second arm.
The observation just mentioned also points to a challenge in extending the analysis of TS for
two-armed bandit to the generalN-armed bandit setting. One might consider analyzing the regret in
theN-armed case by considering only two arms at a time—the first arm and one of the suboptimal
arms. We could use the observation that the probability of playing a suboptimal arm is bounded by
the probability of it exceeding the first arm. However, this probability also depends on the number
of previous plays of the two arms, which in turn depend on the plays of the other arms.  Again,
Auer et al.(2002), in their analysis of UCB algorithm, overcome this difficulty by bounding this
probability forall possible numbers of previous playsof the first arm, and large enough plays of
the suboptimal arm. For Thompson Sampling, due to the observation made earlier, the (distribution
of the) number of previous plays of the first arm needs to be carefully accounted for, which in turn
requires considering all the arms at the same time, thereby leading to a more involved analysis.
Proof outline for two arms setting.Let us first consider the special case of two arms which is
simpler than the generalNarms case.  Firstly, we note that it is sufficient to bound the regret in-
curred during the time stepsafterthe second arm has been playedL= 24(lnT)/∆
## 2
times.  The
expected regret before this event is bounded by24(lnT)/∆because only the plays of the second
arm produce an expected regret of∆; regret is0when the first arm is played.  Next, we observe
that after the second arm has been playedLtimes, the following happens with high probability:
the empirical average reward of the second arm from each play is very close to its actual expected
rewardμ
## 2
, and its beta distribution is tightly concentrated aroundμ
## 2
.  This means that, thereafter,
the first arm would be played at timetifθ
## 1
(t)turns out to be greater than (roughly)μ
## 2
. This obser-
vation allows us to model the number of steps between two consecutive plays of the first arm as a
geometric random variable with parameter close toPr[θ
## 1
(t)> μ
## 2
]. To be more precise, given that
there have beenjplays of the first arm withs(j)successes andf(j) =j−s(j)failures, we want
to estimate the expected number of steps before the first arm is played again (not including the steps
in which the first arm is played).  This is modeled by a geometric random variableX(j,s(j),μ
## 2
## )
with parameterPr[θ
## 1
> μ
## 2
],  whereθ
## 1
has distribution Beta(s(j) + 1,j−s(j) + 1),  and thus
E[X(j,s(j),μ
## 2
## )
s(j)] = 1/Pr[θ
## 1
> μ
## 2
]−1. To bound the overall expected number of steps be-
tween thej
th
and(j+ 1)
th
play of the first arm, we need to take into account the distribution of the
number of successess(j). For largej, we use Chernoff–Hoeffding bounds to say thats(j)/j≈μ
## 1
with high probability, and moreoverθ
## 1
is concentrated around its mean, and thus we get a good esti-
mate ofE[E[X(j,s(j),μ
## 2
## )
s(j)]]. However, for smalljwe do not have such concentration, and it
requires a delicate computation to get a bound onE[E[X(j,s(j),μ
## 2
## )
s(j)]]. The resulting bound
## 39.6

## ANALYSIS OFTHOMPSONSAMPLING
on the expected number of steps between consecutive plays of the first arm bounds the expected
number of plays of the second arm, to yield a good bound on the regret for the two-arms setting.
Proof outline forNarms setting.At any stept, we divide the set of suboptimal arms into two
subsets:saturatedandunsaturated.  The setC(t)of saturated arms at timetconsists of armsa
that have already been played a sufficient number (L
a
= 24(lnT)/∆
## 2
a
) of times, so that with high
probability,θ
a
(t)is tightly concentrated aroundμ
a
.  As earlier, we try to estimate the number of
steps between two consecutive plays of the first arm. Afterj
th
play, the(j+ 1)
th
play of first arm
will occur at the earliest timetsuch thatθ
## 1
(t)> θ
i
(t),∀i6= 1.  The number of steps beforeθ
## 1
## (t)
is greater thanθ
a
(t)of all saturated armsa∈C(t)can be  closely approximated using a geometric
random variable with parameter close toPr(θ
## 1
## ≥max
a∈C(t)
μ
a
), as before. However, even ifθ
## 1
## (t)
is greater than theθ
a
(t)of all saturated armsa∈C(t), it may not get played due to play of an
unsaturated armuwith a greaterθ
u
(t).  Call this event an “interruption” by unsaturated arms.  We
show that if there have beenjplays of first arm withs(j)successes, the expected number of steps
until the(j+ 1)
th
play can be upper bounded by the product of the expected value of a geometric
random variable similar toX(j,s(j),max
a
μ
a
)defined earlier, and the number of interruptions by
the unsaturated arms.  Now, the total number of interruptions by unsaturated arms is bounded by
## ∑
## N
u=2
## L
u
(since an armubecomes saturated afterL
u
plays). The actual number of interruptions is
hard to analyze due to the high variability in the parameters of the unsaturated arms. We derive our
bound assuming the worst case allocation of these
## ∑
u
## L
u
interruptions. This step in the analysis is
the main source of the high exponent of∆in our regret bound for theN-armed case compared to
the two-armed case.
-  Regret bound for the two-armed bandit problem
In this section,  we present a proof of Theorem1,  our result for the two-armed bandit problem.
Recall our assumption that all arms have Bernoulli distribution on rewards, and that the first arm is
the unique optimal arm.
Let random variablej
## 0
denote the number of plays of the first arm untilL=  24(lnT)/∆
## 2
plays of the second arm.  Let random variablet
j
denote the time step at which thej
th
play of the
first arm happens (we definet
## 0
= 0).  Also, let random variableY
j
## =t
j+1
## −t
j
−1measure the
number of time steps between thej
th
and(j+ 1)
th
plays of the first arm (not counting the steps in
which thej
th
and(j+ 1)
th
plays happened), and lets(j)denote the number of successes in the first
jplays of the first arm. Then the expected number of plays of the second arm in timeTis bounded
by
## E[k
## 2
## (T)]≤L+E
## [
## ∑
## T−1
j=j
## 0
## Y
j
## ]
## .
To understand the expectation ofY
j
, it will be useful to define another random variableX(j,s,y)
as follows. We perform the following experiment until it succeeds: check if a Beta(s+ 1,j−s+ 1)
distributed  random  variable  exceeds  a  thresholdy.   For  each  experiment,  we  generate  the  beta-
distributed r.v. independently of the previous ones. Now defineX(j,s,y)to be the number of trials
beforethe experiment succeeds.  Thus,X(j,s,y)takes non-negative integer values, and is a geo-
metric random variable with parameter (success probability)1−F
beta
s+1,j−s+1
(y). HereF
beta
α,β
denotes
the cdf of the beta distribution with parametersα,β.  Also, letF
## B
n,p
denote the cdf of thebinomial
distribution with parameters(n,p).
We will relateYandXshortly.  The following lemma provides a handle on the expectation of
## X.
## 39.7

## AGRAWALGOYAL
Lemma 4For all non-negative integersj,s≤j, and for ally∈[0,1],
E[X(j,s,y)] =
## 1
## F
## B
j+1,y
## (s)
## −1,
whereF
## B
n,p
denotes the cdf of the binomial distribution with parameters(n,p).
ProofBy  the  well-known  formula  for  the  expectation  of  a  geometric  random  variable  and  the
definition ofXwe have,E[X(j,s,y)]  =
## 1
## 1−F
beta
s+1,j−s+1
## (y)
−1(The additive−1is there because
we do not count the final step where the Beta r.v.  is greater thany.)  The lemma then follows from
Fact1 in AppendixB.
Recall thatY
j
was defined as the number of steps beforeθ
## 1
(t)> θ
## 2
(t)happens for the first time after
thej
th
play of the first arm.  Now, consider the number of steps beforeθ
## 1
(t)> μ
## 2
## +
## ∆
## 2
happens
for the first time after thej
th
play of the first arm.  Givens(j), this has the same distribution as
## X(j,s(j),μ
## 2
## +
## ∆
## 2
). However,Y
j
can be larger than this number if (and only if) at some time stept
betweent
j
andt
j+1
## ,θ
## 2
(t)> μ
## 2
## +
## ∆
## 2
. In that case we use the fact thatY
j
is always bounded byT.
Thus, for anyj≥j
## 0
, we can boundE[Y
j
## ]as,
## E[Y
j
]≤E[min{X(j,s(j),μ
## 2
## +
## ∆
## 2
## ),T}] +E[
## ∑
t
j+1−1
t=t
j
## +1
T·I(θ
## 2
(t)> μ
## 2
## +
## ∆
## 2
## )].
Here notationI(E)is the indicator for eventE, i.e., its value is1if eventEhappens and0otherwise.
In the first term of RHS, the expectation is over distribution ofs(j)as well as over the distribution
of the geometric variableX(j,s(j),μ
## 2
## +
## ∆
## 2
). Since we are interested only inj≥j
## 0
, we will instead
use the similarly obtained bound onE[Y
j
·I(j≥j
## 0
## )],
## E[Y
j
·I(j≥j
## 0
)]≤E[min{X(j,s(j),μ
## 2
## +
## ∆
## 2
## ),T}] +E[
## ∑
t
j+1−1
t=t
j
## +1
T·I(θ
## 2
(t)> μ
## 2
## +
## ∆
## 2
)·I(j≥j
## 0
## )].
This gives,
## E[
## ∑
## T−1
j=j
## 0
## Y
j
## ]≤
## ∑
## T−1
j=0
E[min{X(j,s(j),μ
## 2
## +
## ∆
## 2
## ),T}] +T·
## ∑
## T−1
j=0
## E[
## ∑
t
j+1−1
t=t
j
## +1
## I(θ
## 2
(t)> μ
## 2
## +
## ∆
## 2
## ,j≥j
## 0
## )]
## ≤
## ∑
## T−1
j=0
E[min{X(j,s(j),μ
## 2
## +
## ∆
## 2
## ),T}] +T·
## ∑
## T
t=1
## Pr(θ
## 2
(t)> μ
## 2
## +
## ∆
## 2
## ,k
## 2
(t)≥L).
The last inequality holds because for anyt∈[t
j
## + 1,t
j+1
## −1],j≥j
## 0
, by definitionk
## 2
(t)≥L.
We denote the event{θ
## 2
## (t)≤μ
## 2
## +
## ∆
## 2
ork
## 2
(t)< L}byE
## 2
(t).  In words, this is the event that if
sufficient number of plays of second arm have happened until timet, thenθ
## 2
(t)is not much larger
thanμ
## 2
; intuitively, we expect this event to be a high probability event as we will show.
## E
## 2
(t)is the
event{θ
## 2
(t)> μ
## 2
## +
## ∆
## 2
andk
## 2
(t)≥L}used in the above equation. Next, we boundPr(E
## 2
## (t))and
E[min{X(j,s(j),μ
## 2
## +
## ∆
## 2
## ),T}].
Lemma 5∀t,Pr(E
## 2
## (t))≥1−
## 2
## T
## 2
## .
ProofRefer to AppendixC.1.
Lemma 6Consider any positivey < μ
## 1
, and let∆
## ′
## =μ
## 1
−y. Also, letR=
μ
## 1
## (1−y)
y(1−μ
## 1
## )
>1, and let
Ddenote the KL-divergence betweenμ
## 1
andy, i.e.D=yln
y
μ
## 1
+ (1−y) ln
## 1−y
## 1−μ
## 1
## .
E[E[min{X(j,s(j),y), T}
s(j)]]≤
## 
## 
## 
## 
## 
## 
## 
## 
## 
## 
## 
## 1 +
## 2
## 1−y
## +
μ
## 1
## ∆
## ′
e
−Dj
j <
y
## D
lnR,
## 1 +
## R
y
## 1−y
e
−Dj
## +
μ
## 1
## ∆
## ′
e
−Dj
y
## D
lnR≤j <
4 lnT
## ∆
## ′2
## ,
## 16
## T
j≥
4 lnT
## ∆
## ′2
## ,
## 39.8

## ANALYSIS OFTHOMPSONSAMPLING
where the outer expectation is taken overs(j)distributed as Binomial(j,μ
## 1
## ).
ProofThe complete proof of this lemma is included in AppendixC.2; here we provide some high
level ideas.
Using Lemma4, the expected value ofX(j,s(j),y)for any givens(j),
E[X(j,s(j),y)
s(j)] =
## 1
## F
## B
j+1,y
## (s(j))
## −1.
For largej, i.e.,j≥4(lnT)/∆
## ′2
, we use Chernoff–Hoeffding bounds to argue that with probability
at least (1−
## 8
## T
## 2
),s(j)will be greater thanμ
## 1
j−∆
## ′
j/2.  And, fors(j)≥μ
## 1
j−∆
## ′
j/2 =yj+
## ∆
## ′
j/2, we can show that the probabilityF
## B
j+1,y
(s(j))will be at least1−
## 8
## T
## 2
, again using Chernoff–
Hoeffding bounds.  These observations allow us to derive thatE[E[min{X(j,s(j),y),T}]]≤
## 16
## T
## ,
forj≥4(lnT)/∆
## ′2
## .
For smallj, the argument is more delicate.  In this case,s(j)could be small with a significant
probability.  More precisely,s(j)could take a valuessmaller thanyjwith binomial probability
f
## B
j,μ
## 1
(s).   For  suchs,  we  use  the  lower  boundF
## B
j+1,y
(s)≥(1−y)F
## B
j,y
(s) +yF
## B
j,y
## (s−1)≥
(1−y)F
## B
j,y
## (s)≥(1−y)f
## B
j,y
(s), and then bound the ratiof
## B
j,μ
## 1
## (s)/f
## B
j,y
(s)in terms of∆
## ′
,Rand
KL-divergenceD.  Fors(j) =s≥ dyje, we use the observation that sincedyjeis greater than or
equal to the median of Binomial(j,y)(seeJogdeo and Samuels(1968)), we haveF
## B
j,y
## (s)≥1/2.
After some algebraic manipulations, we get the result of the lemma.
Using Lemma 5, and Lemma6 fory=μ
## 2
+ ∆/2, and∆
## ′
=  ∆/2, we can bound the expected
number of plays of the second arm as:
## E[k
## 2
## (T)]   =L+E
## [
## ∑
## T−1
j=j
## 0
## Y
j
## ]
## ≤L+
## ∑
## T−1
j=0
## E
## [
## E
## [
min{X(j,s(j),μ
## 2
## +
## ∆
## 2
),T}s(j)
## ] ]
## +
## ∑
## T
t=1
T·Pr(E
## 2
## (t))
## ≤L+
4 lnT
## ∆
## ′2
## +
## ∑
4(lnT)/∆
## ′2
## −1
j=0
μ
## 1
## ∆
## ′
e
−Dj
## +
## (
y
## D
lnR
## )
## 2
## 1−y
## +
## ∑
4(lnT)/∆
## ′2
## −1
j=
y
## D
lnR
## R
y
e
−Dj
## 1−y
## +
## 16
## T
## ·T+ 2
## ≤
40 lnT
## ∆
## 2
## +
## 48
## ∆
## 4
## + 18,
## (1)
where the last inequality is obtained after some algebraic manipulations;  details are provided in
## Appendix C.3.
This gives a regret bound of
E[R(T)] =E[∆·k
## 2
## (T)]≤
## (
40 lnT
## ∆
## +
## 48
## ∆
## 3
## + 18∆
## )
## .
-  Regret bound for theN-armed bandit problem
In this section, we prove Theorem2, our result for theN-armed bandit problem. Again, we assume
that all arms have Bernoulli distribution on rewards, and that the first arm is the unique optimal arm.
At every time stept, we divide the set of suboptimal arms into saturated and unsaturated arms.
We  say  that  an  armi6=  1is  in  the  saturated  setC(t)at  timet,  if  it  has  been  played  at  least
## L
i
## :=
24 lnT
## ∆
## 2
i
times before timet.  We bound the regret due to playing unsaturated and saturated
suboptimal  arms  separately.   The  former  is  easily  bounded  as  we  will  see;  most  of  the  work  is
## 39.9

## AGRAWALGOYAL
Figure 1:  IntervalI
j
in  bounding  the  latter.   For  this,  we  bound  the  number  of  plays  of  saturated  arms  between  two
consecutive plays of the first arm.
In the following, by an interval of time we mean a set of contiguous time steps.  Let r.v.I
j
denote the interval between (and excluding) thej
th
and(j+ 1)
th
plays of the first arm. We say that
eventM(t)holds at timet, ifθ
## 1
## (t)exceedsμ
i
## +
## ∆
i
## 2
of all the saturated arms, i.e.,
## M(t) :θ
## 1
## (t)>max
i∈C(t)
μ
i
## +
## ∆
i
## 2
## .(2)
Fortsuch thatC(t)is empty, we defineM(t)to hold trivially.
Let r.v.γ
j
denote the number of occurrences of eventM(t)in intervalI
j
## :
γ
j
=|{t∈I
j
:M(t) = 1}|.(3)
EventsM(t)divideI
j
into sub-intervals in a natural way: For`= 2toγ
j
, let r.v.I
j
(`)denote the
sub-interval ofI
j
between the(`−1)
th
and`
th
occurrences of eventM(t)inI
j
(excluding the time
steps in which the eventM(t)occurs).  We also defineI
j
(1)andI
j
## (γ
j
## + 1): Ifγ
j
>0thenI
j
## (1)
denotes the sub-interval inI
j
before the first occurrence of eventM(t)inI
j
; andI
j
## (γ
j
## + 1)denotes
the sub-interval inI
j
after the last occurrence of eventM(t)inI
j
## . Forγ
j
= 0we haveI
j
## (1) =I
j
## .
Figure 1 shows an example of intervalI
j
along with sub-intervalsI
j
(`); in this figureγ
j
## = 4.
Let us define eventE(t)as
## E(t) :{θ
i
## (t)∈[μ
i
## −∆
i
## /2,μ
i
## + ∆
i
/2],∀i∈C(t)}.
In words,E(t)denotes the event that all saturated arms haveθ
i
(t)tightly concentrated around their
means.  Intuitively, from the definition of saturated arms,E(t)should hold with high probability;
we prove this in the lemma below.
Lemma 7For allt,Pr(E(t))≥1−
## 4(N−1)
## T
## 2
## .
Also, for allt,j, ands≤j,Pr(E(t)|s(j) =s)≥1−
## 4(N−1)
## T
## 2
## .
## 39.10

## ANALYSIS OFTHOMPSONSAMPLING
ProofRefer to AppendixC.4.
The stronger bound given by the second statement of lemma above will be useful later in the proof.
Observe that since a saturated armican be played at a steptonly ifθ
i
(t)is greater thanθ
## 1
## (t),
the saturated armican be played at a time steptwhereM(t)holds only ifθ
i
(t)> μ
i
## + ∆
i
## /2.
Thus, unless the high probability eventE(t)is violated,M(t)denotes a play of an unsaturated arm
at timet, andγ
j
essentially denotes the number of plays of unsaturated arms in intervalI
j
## .  And,
the number of plays of saturated arms in intervalI
j
is at most
## ∑
γ
j
## +1
## `=1
## |I
j
## (`)|+
## ∑
t∈I
j
## I(
## E(t)).
We are interested in bounding regret due to playing saturated arms, which depends not only on
the number of plays, but also onwhichsaturated arm is played at each time step.  LetV
## `,a
j
denote
the number of steps inI
j
(`), for whichais the best saturated arm, i.e.
## V
## `,a
j
=|{t∈I
j
## (`) :μ
a
= max
i∈C(t)
μ
i
## }|,(4)
(resolve the ties for best saturated arm using an arbitrary, but fixed, ordering on arms).   In Figure
1, we illustrate this notation by showing steps{V
## 4,a
j
}for intervalI
j
(4). In the example shown, we
assume thatμ
## 1
> μ
## 2
>···> μ
## 6
, and that the suboptimal arms got added to the saturated setC(t)
in order5,3,4,2,6, so that initially5is the best saturated arm, then3is the best saturated arm, and
finally2is the best saturated arm.
Recall thatM(t)holds trivially for alltsuch thatC(t)is empty. Therefore, there is at least one
saturated arm at allt∈I
j
(`), and henceV
## `,a
j
,a= 2,...,Nare well defined and cover the interval
## I
j
## (`),
## |I
j
## (`)|=
## ∑
## N
a=2
## V
## `,a
j
## .
Next,  we  will  show  that  the  regret  due  to  playing  any  saturated  arm  at  a  time  steptin  one  of
theV
## `,a
j
steps is at most3∆
a
## +I(
E(t)).   The idea is that if all saturated arms have theirθ
i
## (t)
tightly concentrated around their meansμ
i
, then either the arm with the highest mean (i.e., the best
saturated arma) or an arm with mean very close toμ
a
will be chosen to be played during theseV
## `,a
j
steps.  That is, if a saturated armiis played at a timetamong one of theV
## `,a
j
steps, then, either
E(t)is violated, i.e.θ
i
## ′
(t)for some saturated armi
## ′
is not close to its mean, or
μ
i
## + ∆
i
## /2≥θ
i
## (t)≥θ
a
## (t)≥μ
a
## −∆
a
## /2,
which implies that
## ∆
i
## =μ
## 1
## −μ
i
## ≤μ
## 1
## −μ
a
## +
## ∆
a
## 2
## +
## ∆
i
## 2
## ⇒∆
i
## ≤3∆
a
## .(5)
Therefore, regret due to play of a saturated arm at a timetin one of theV
## `,a
j
steps is at most
## 3∆
a
## +I(
E(t)). With slight abuse of notation let us uset∈V
## `,a
j
to indicate thattis one of theV
## `,a
j
steps inI
j
(`). Then, the expected regretdue to playing saturated armsin intervalI
j
is bounded as
## E[R
s
## (I
j
## )]≤E
## [
## ∑
γ
j
## +1
## `=1
## ∑
## N
a=2
## ∑
t∈V
## `,a
j
## (3∆
a
## +I(
## E(t)))
## ]
## +
## ∑
t∈I
j
## I(
## E(t)).
## =E
## [
## ∑
γ
j
## +1
## `=1
## ∑
## N
a=2
## 3∆
a
## V
## `,a
j
## ]
## + 2E
## [
## ∑
t∈I
j
I(E(t))
## ]
## .(6)
The second term in above will be bounded using Lemma 7. For bounding the first term, we establish
the following lemma.
## 39.11

## AGRAWALGOYAL
Lemma 8For allj,
## E
## [
## ∑
γ
j
## +1
## `=1
## ∑
a
## V
## `,a
j
## ∆
a
## ]
## ≤E
## [
## E[(γ
j
## + 1)
s(j)]
## ∑
## N
a=2
## ∆
a
## E
## [
min{X(j,s(j),μ
a
## +
## ∆
a
## 2
),T}s(j)
## ]
## ]
## (7)
ProofThe key observation used in proving this lemma is that given   a fixed value ofs(j)  =s,
the random variableV
## `,a
j
is stochastically dominated by random variableX(j,s,μ
a
## +
## ∆
a
## 2
## )(defined
earlier as a geometric variable denoting the number of trials before an independent sample from
Beta(s+ 1,j−s+ 1)distribution exceedsμ
a
## +
## ∆
a
## 2
). A technical difficulty in deriving the inequal-
ity above is that the random variablesγ
j
andV
## `,a
j
are not independent in general (both depend on
the values taken by{θ
i
(t)}over the interval). This issue is handled through careful conditioning of
the random variables on history. The details of the proof are provided in AppendixC.5.
Next we illustrate the main ideas of the remaining proof by proving a weaker bound of
## (
## ∑
i
logT
## ∆
## 2
i
## )
## 2
on the expected regret. The proof of the bound(logT)
## (
## ∑
i
## 1
## ∆
## 2
i
## )
## 2
of Theorem 2requires a slightly
more careful analysis of this part, the complete details are given in AppendixD.
Consider the regret due to playing saturated arms until
## ∑
## N
i=2
## L
i
plays of the first arm.  After
these many plays, the first arm will be concentrated enough so that the probability of playing any
saturated arm (and hence the regret) will be very small. Now, using Lemma8, the regret contributed
by the first term in (6) can be loosely bounded by
## 3E
## [
## ∑
## ∑
i
## L
i
j=0
## E[(γ
j
## + 1)
s(j)]
## ∑
a
## ∆
a
E[min{X(j,s(j),y
a
## ),T}
s(j)]
## ]
## ≤3E
## [(
## ∑
## ∑
i
## L
i
j=0
## E[(γ
j
## + 1)
s(j)]
## )(
## ∑
## ∑
i
## L
i
j=0
## ∑
a
## ∆
a
E[min{X(j,s(j),y
a
## ),T}
s(j)]
## )]
## .
Recall thatγ
j
is (approximately) the total number of plays of unsaturated arms in intervalI
j
## . There-
fore, the first term in the product above is bounded by the total number of plays of unsaturated arms,
i.e.O(
## ∑
## N
i=2
## L
i
). For the second term, using Lemma6, we observe thatE[E[min{X(j,s(j),y
a
## ),T}
s(j)]]
is bounded byO(
## 1
## ∆
a
). Therefore, the second term is bounded byO(
## ∑
## N
i=2
## L
i
)as well. This gives
a bound ofO((
## ∑
i
## L
i
## )
## 2
## ) =O(
## (
## ∑
i
logT
## ∆
## 2
i
## )
## 2
)on the above, and thus on the contribution of the first
term of (6) towards the regret.  The total contribution of the second term in Equation (6) can be
bounded by a constant using Lemma7.
Since an unsaturated armubecomes saturated afterL
u
plays, regret due to unsaturated arms is at
most
## ∑
## N
u=2
## L
u
## ∆
u
= 24(lnT)
## (
## ∑
## N
u=2
## 1
## ∆
u
## )
.  Summing the regret due to saturated and unsaturated
arms, we obtain the weaker bound ofO((
## ∑
i
logT
## ∆
## 2
i
## )
## 2
)on regret. For details of the proof of the tighter
bound of Theorem2, see appendixD.
Conclusion.In this paper,  we showed theoretical guarantees for Thompson Sampling close to
other state of the art methods, like UCB. Our result is a first step in theoretical understanding of
TS. With further work, we hope that our techniques in this paper will be useful in providing several
extensions, including a tighter analysis of the regret bound to close the gap between our upper bound
and the lower bound ofLai and Robbins(1985), analysis of TS for delayed and batched feedbacks,
contextual bandits, prior mismatch and posterior reshaping discussed inChapelle and Li(2011).
## 39.12

## ANALYSIS OFTHOMPSONSAMPLING
## References
P. Auer, N. Cesa-Bianchi, and P. Fischer.  Finite-time analysis of the multiarmed bandit problem.
## Machine Learning, 47(2-3):235–256, 2002.
O. Chapelle and L. Li. An empirical evaluation of thompson sampling. InNIPS, 2011.
A. Garivier and O. Capp
## ́
e.  The KL-UCB algorithm for bounded stochastic bandits and beyond.  In
Conference on Learning Theory (COLT), 2011.
J. C. Gittins.Multi-armed Bandit Allocation Indices.   Wiley Interscience Series in Systems and
Optimization. John Wiley and Son, 1989.
T. Graepel,  J. Q. Candela,  T. Borchert,  and R. Herbrich.   Web-scale bayesian click-through rate
prediction for sponsored search advertising in microsoft’s bing search engine.  InICML, pages
## 13–20, 2010.
O.-C. Granmo. Solving two-armed bernoulli bandit problems using a bayesian learning automaton.
International Journal of Intelligent Computing and Cybernetics (IJICC), 3(2):207–234, 2010.
K. Jogdeo and S. M. Samuels.  Monotone Convergence of Binomial Probabilities and A General-
ization of Ramanujan’s equation.The Annals of Mathematical Statistics, (4):1191–1195, 1968.
## E. Kaufmann, O. Capp
## ́
e, and A. Garivier.  On bayesian upper confidence bounds for bandit prob-
lems.   InFifteenth International Conference on Artificial Intelligence and Statistics (AISTAT),
## 2012.
T. L. Lai and H. Robbins.  Asymptotically efficient adaptive allocation rules.Advances in Applied
## Mathematics, 6:4–22, 1985.
O.-A. Maillard, R. Munos, and G. Stoltz. Finite-time analysis of multi-armed bandits problems with
kullback-leibler divergences. InConference on Learning Theory (COLT), 2011.
B. C. May and D. S. Leslie. Simulation studies in optimistic bayesian sampling in contextual-bandit
problems.  Technical Report 11:02, Statistics Group, Department of Mathematics, University of
## Bristol, 2011.
B. C. May, N. Korda, A. Lee, and D. S. Leslie.  Optimistic bayesian sampling in contextual-bandit
problems.  Technical Report 11:01, Statistics Group, Department of Mathematics, University of
## Bristol, 2011.
S. Scott. A modern bayesian look at the multi-armed bandit.Applied Stochastic Models in Business
and Industry, 26:639–658, 2010.
W. R. Thompson.  On the likelihood that one unknown probability exceeds another in view of the
evidence of two samples.Biometrika, 25(3-4):285–294, 1933.
## 39.13

## AGRAWALGOYAL
Appendix A.  Multiple optimal arms
Consider theN-armed bandit problem withμ
## ∗
= max
i
μ
i
.  We will show that adding another arm
with expected rewardμ
## ∗
can only decrease the expected regret of TS algorithm.  Suppose that we
added armN+ 1with expected rewardμ
## ∗
.  Consider the expected regret for the new bandit in
timeT, conditioned on the exact time steps among1,...,T, on which armN+ 1is played by the
algorithm. Since the armN+ 1has expected rewardμ
## ∗
, there is no regret in these time steps. Now
observe that in the remaining time steps, the algorithm behaves exactly as it would for the original
bandit withNarms. Therefore, given that the(N+ 1)
th
arm is playedxtimes, the expected regret
in timeTfor the new bandit will be same as the expected regret in timeT−xfor the original
bandit.  LetR
## N
(T)andR
## N+1
(T)denote the expected regret in timeTfor the original and new
bandit, respectively. Then,
## E
## [
## R
## N+1
## (T)
## ]
## =E
## [
## E
## [
## R
## N+1
## (T)
k
## N+1
## (T)
## ]]
## =E
## [
## E
## [
## R
## N
(T−k
## N+1
(T))k
## N+1
## (T)
## ]]
## ≤E
## [
## E
## [
## R
## N
(T)k
## N+1
## (T)
## ]]
## =E
## [
## R
## N
## (T)
## ]
## .
This  argument  shows  that  the  expected  regret  of  Thompson  Sampling  for  theN-armed  bandit
problem withroptimal arms is bounded by the expected regret of Thompson Sampling for the
(N−r+ 1)-armed bandit problem obtained on removing (any)r−1of the optimal arms.
Appendix B.  Facts used in the analysis
## Fact 1
## F
beta
α,β
(y) = 1−F
## B
α+β−1,y
## (α−1),
for all positive integersα,β.
ProofThis fact is well-known (it’s mentioned on Wikipedia) but we are not aware of a specific
reference. Since the proof is easy and short we will present a proof here. The Wikipedia page also
mentions that it can be proved using integration by parts.  Here we provide a direct combinatorial
proof which may be new.
One well-known way to generate a r.v.  with cdfF
beta
α,β
for integerαandβis the following:
generate uniform in[0,1]r.v.sX
## 1
## ,X
## 2
## ,...,X
α+β−1
independently.  Let the values of these r.v.  in
sorted increasing order be denotedX
## ↑
## 1
## ,X
## ↑
## 2
## ,...,X
## ↑
α+β−1
.  ThenX
## ↑
α
has cdfF
beta
α,β
.  ThusF
beta
α,β
## (y)
is the probability thatX
## ↑
α
## ≤y.
We now reinterpret this probability using the binomial distribution: The eventX
## ↑
α
## ≤yhappens
iff for at leastαof theX
## 1
## ,...,X
α+β−1
we haveX
i
≤y.  For eachX
i
we havePr[X
i
## ≤y] =y;
thus the probability that for at mostα−1of theX
i
’s we haveX
i
≤yisF
## B
α+β−1,y
(α−1). And so
the probability that for at leastαof theX
i
’s we haveX
i
≤yis1−F
## B
α+β−1,y
## (α−1).
The median of an integer-valued random variableXis an integermsuch thatPr(X≤m)≥
1/2andPr(X≥m)≥1/2. The following fact says that the median of the binomial distribution is
close to its mean.
Fact 2 (Jogdeo and Samuels(1968))Median of the binomial distribution Binomial(n,p)is either
bnpcordnpe.
## 39.14

## ANALYSIS OFTHOMPSONSAMPLING
Fact 3 ((Chernoff–Hoeffding bounds))LetX
## 1
## ,...,X
n
be random variables with common range
[0,1]and such thatE[X
t
## X
## 1
## ,...,X
t−1
] =μ. LetS
n
## =X
## 1
## +...+X
n
. Then for alla≥0,
Pr(S
n
## ≥nμ+a)≤e
## −2a
## 2
## /n
## ,
Pr(S
n
## ≤nμ−a)≤e
## −2a
## 2
## /n
## .
Lemma 9For alln,p∈[0,1],δ≥0,
## F
## B
n,p
## (np−nδ)≤e
## −2nδ
## 2
## ,1−F
## B
n,p
## (np+nδ)≤e
## −2nδ
## 2
## ,(8)
## 1−F
## B
n+1,p
## (np+nδ)≤
e
## 4δ
e
## 2nδ
## 2
## .(9)
ProofThe first result is a simple application of Chernoff–Hoeffding bounds from Fact3.  For the
second result, we observe that,
## F
## B
n+1,p
(np+nδ) = (1−p)F
## B
n,p
(np+nδ) +pF
## B
n,p
(np+nδ−1)≥F
## B
n,p
## (np+nδ−1).
By Chernoff–Hoeffding bounds,
## 1−F
## B
n,p
## (np+δn−1)≤e
## −2(δn−1)
## 2
## /n
## =e
## −2(n
## 2
δ
## 2
## +1−2δn)/n
## ≤e
## −2nδ
## 2
## +4δ
## =
e
## 4δ
e
## 2nδ
## 2
## .
Appendix C.  Proofs of Lemmas
C.1.  Proof of Lemma5
ProofIn this lemma, we lower bound the probability ofE
## 2
## (t)by1−
## 2
## T
## 2
. Recall that eventE
## 2
## (t)
holds if the following is true:
## {θ
## 2
## (t)≤μ
## 2
## +
## ∆
## 2
## }or{k
## 2
## (t)< L}.
Also defineA(t)as the event
## A(t) :
## S
## 2
## (t)
k
## 2
## (t)
## ≤μ
## 2
## +
## ∆
## 4
## ,
whereS
## 2
## (t),k
## 2
(t)denote the number of successes and number of plays respectively of the second
arm until timet−1. We will upper bound the probability ofPr(
## E
## 2
(t)) = 1−Pr(E
## 2
## (t))as:
## Pr(
## E
## 2
## (t))   =   Pr(θ
## 2
## (t)≥μ
## 2
## +
## ∆
## 2
## ,k
## 2
(t)≥L)
≤Pr(
## A(t),k
## 2
(t)≥L) + Pr(θ
## 2
## (t)≥μ
## 2
## +
## ∆
## 2
## ,k
## 2
(t)≥L,A(t)).(10)
## 39.15

## AGRAWALGOYAL
For clarity of exposition,  let us define another random variableZ
## 2,M
,  as the average number of
successes  over  the  firstMplays  of  the  second  arm.   More  precisely,  let  random  variableZ
## 2,m
denote the output of them
th
play of the second arm. Then,
## Z
## 2,M
## =
## 1
## M
## M
## ∑
m=1
## Z
## 2,m
## .
Note that by definition,
## Z
## 2,k
## 2
## (t)
## =
## S
## 2
## (t)
k
## 2
## (t)
.  Also,Z
## 2,M
is the average ofMiid Bernoulli variables,
each with meanμ
## 2
## .
Now, for allt,
Pr(A(t),k
## 2
(t)≥L)   =
## ∑
## T
## `=L
## Pr(
## Z
## 2,k
## 2
## (t)
## ≥μ
## 2
## +
## ∆
## 4
## ,k
## 2
## (t) =`)
## =
## ∑
## T
## `=L
## Pr(
## Z
## 2,`
## ≥μ
## 2
## +
## ∆
## 4
## ,k
## 2
## (t) =`)
## ≤
## ∑
## T
## `=L
## Pr(
## Z
## 2,`
## ≥μ
## 2
## +
## ∆
## 4
## )
## ≤
## ∑
## T
## `=L
e
## −2`∆
## 2
## /16
## ≤
## 1
## T
## 2
## .
The second last inequality is by applying Chernoff bounds, sinceZ
## 2,`
is simply the average of`iid
Bernoulli variables each with meanμ
## 2
## .
We will derive the bound on second probability term in (10) in a similar manner. It will be useful to
defineW(`,z)as a random variable distributed as Beta(`z+ 1,`−`z+ 1). Note that if at timet, the
number of plays of second arm isk
## 2
(t) =`, thenθ
## 2
(t)is distributed as Beta(`Z
## 2,`
## +1,`−`Z
## 2,`
## +1),
i.e. same asW(`,Z
## 2,`
## ).
## Pr(θ
## 2
(t)> μ
## 2
## +
## ∆
## 2
,A(t),k
## 2
(t)≥L)   =
## T
## ∑
## `=L
## Pr(θ
## 2
(t)> μ
## 2
## +
## ∆
## 2
,A(t),k
## 2
## (t) =`)
## ≤
## T
## ∑
## `=L
## Pr(θ
## 2
## (t)>
## S
## 2
## (t)
k
## 2
## (t)
## −
## ∆
## 4
## +
## ∆
## 2
## ,k
## 2
## (t) =`)
## =
## T
## ∑
## `=L
Pr(W(`,
## Z
## 2,`
## )>Z
## 2,`
## +
## ∆
## 4
## ,k
## 2
## (t) =`)
## ≤
## T
## ∑
## `=L
Pr(W(`,Z
## 2,`
## )>Z
## 2,`
## +
## ∆
## 4
## )
## (using Fact 1)=
## T
## ∑
## `=L
## E
## [
## F
## B
## `+1,
## Z
## 2,`
## +
## ∆
## 4
## (`Z
## 2,`
## )
## ]
## ≤
## T
## ∑
## `=L
## E
## [
## F
## B
## `,Z
## 2,`
## +
## ∆
## 4
## (`Z
## 2,`
## )
## ]
## ≤
## T
## ∑
## `=L
exp{−
## 2∆
## 2
## `
## 2
## /16
## `
## }
≤Te
## −2L∆
## 2
## /16
## =
## 1
## T
## 2
## .
## 39.16

## ANALYSIS OFTHOMPSONSAMPLING
The third-last inequality follows from the observation that
## F
## B
n+1,p
(r) = (1−p)F
## B
n,p
(r) +pF
## B
n,p
(r−1)≤(1−p)F
## B
n,p
(r) +pF
## B
n,p
(r) =F
## B
n,p
## (r).
And,  the  second-last  inequality  follows  from  Chernoff–Hoeffding  bounds  (refer  to  Fact3  and
## Lemma9).
C.2.  Proof of Lemma6
ProofUsing Lemma4, the expected value ofX(j,s(j),y)for any givens(j),
E[X(j,s(j),y)
s(j)] =
## 1
## F
## B
j+1,y
## (s(j))
## −1.
Case of largej:First, we consider the case of largej, i.e.  whenj≥4(lnT)/∆
## ′2
.  Then, by
simple application of Chernoff–Hoeffding bounds (refer to Fact3 and Lemma9), we can derive
that for anys≥(y+
## ∆
## ′
## 2
## )j,
## F
## B
j+1,y
(s)≥F
## B
j+1,y
## (yj+
## ∆
## ′
j
## 2
## )≥1−
e
## 4∆
## ′
## /2
e
## 2j∆
## ′2
## /4
## ≥1−
e
## 2∆
## ′
## T
## 2
## ≥1−
## 8
## T
## 2
## ,
giving that fors≥y(j+
## ∆
## ′
## 2
),E[X(j+ 1,s,y)]≤
## 1
## (1−
## 8
## T
## 2
## )
## −1.
Again using Chernoff–Hoeffding bounds,  the probability thats(j)takes values smaller than
## (y+
## ∆
## ′
## 2
)jcan be bounded as,
## F
## B
j,μ
## 1
## (yj+
## ∆
## ′
j
## 2
## ) =F
## B
j,μ
## 1
## (μ
## 1
j−
## ∆
## ′
j
## 2
## )≤e
## −2j
## ∆
## ′2
## 4
## ≤
## 1
## T
## 2
## <
## 8
## T
## 2
## .
For these values ofs(j), we will use the upper bound ofT. Thus,
E[min{E[X(j,s(j),y)s(j)],T}]≤(1−8/T
## 2
## )·
## (
## 1
## (1−8/T
## 2
## )
## −1
## )
## +
## 8
## T
## 2
## ·T≤
## 16
## T
## .
Case of smallj:For smallj, the argument is more delicate. We use,
E[E[X(j,s(j),y)
s(j)]] =E
## [
## 1
## F
## B
j+1,y
## (s(j))
## −1
## ]
## =
j
## ∑
s=0
f
## B
j,μ
## 1
## (s)
## F
## B
j+1,y
## (s)
## −1,(11)
wheref
## B
j,μ
## 1
denotes pdf of the Binomial(j,μ
## 1
)distribution.  We use the observation that fors≥
dy(j+ 1)e,F
## B
j+1,y
(s)≥1/2. This is because the median of a Binomial(n,p)distribution is either
bnpcordnpe(seeJogdeo and Samuels(1968)). Therefore,
j
## ∑
s=dy(j+1)e
f
## B
j,μ
## 1
## (s)
## F
## B
j+1,y
## (s)
## ≤2.(12)
## 39.17

## AGRAWALGOYAL
For smalls, i.e.,s≤ byjc, we useF
## B
j+1,y
(s) = (1−y)F
## B
j,y
(s) +yF
j,y
(s−1)≥(1−y)F
## B
j,y
## (s)
andF
## B
j,y
## (s)≥f
## B
j,y
(s), to get
byjc
## ∑
s=0
f
## B
j,μ
## 1
## (s)
## F
## B
j+1,y
## (s)
## ≤
byjc
## ∑
s=0
## 1
## (1−y)
f
## B
j,μ
## 1
## (s)
f
## B
j,y
## (s)
## =
byjc
## ∑
s=0
## 1
## (1−y)
μ
s
## 1
## (1−μ
## 1
## )
j−s
y
s
## (1−y)
j−s
## =
byjc
## ∑
s=0
## 1
## (1−y)
## R
s
## (1−μ
## 1
## )
j
## (1−y)
j
## =
## 1
## (1−y)
## (
## R
byjc+1
## −1
## R−1
## )
## (1−μ
## 1
## )
j
## (1−y)
j
## ≤
## 1
## (1−y)
## R
## R−1
μ
yj
## 1
## (1−μ
## 1
## )
## (j−yj)
y
yj
## (1−y)
j−yj
## =
μ
## 1
μ
## 1
## −y
e
−Dj
## =
μ
## 1
## ∆
## ′
e
−Dj
## .(13)
Ifbyjc<dyje<dy(j+ 1)e, then we need to additionally considers=dyje. Note, however,
that in this casedyje≤yj+y. Fors=dyje,
f
## B
j,μ
## 1
## (s)
## F
## B
j+1,y
## (s)
## ≤
## 1
(1−y)F
## B
j,y
## (s)
## ≤
## 2
## 1−y
## .(14)
Alternatively, we can use the following bound fors=dyje,
f
## B
j,μ
## 1
## (s)
## F
## B
j+1,y
## (s)
## ≤
## 1
## (1−y)
f
## B
j,μ
## 1
## (s)
## F
## B
j,y
## (s)
## ≤
## 1
## (1−y)
f
## B
j,μ
## 1
## (s)
f
## B
j,y
## (s)
## ≤
## 1
## (1−y)
## R
s
## (
## 1−μ
## 1
## 1−y
## )
j
## ≤
## 1
## (1−y)
## R
yj+y
## (
## 1−μ
## 1
## 1−y
## )
j
## (becauses=dyje≤yj+y)
## ≤
## R
y
## (1−y)
e
−Dj
## .(15)
Next, we substitute the bounds from (12)-(15) in Equation (11) to get the result in the lemma.
In this substitution, fors=dyje, we use the bound in Equation (14) whenj <
y
## D
lnR, and the
bound in Equation (15) whenj≥
y
## D
lnR.
## 39.18

## ANALYSIS OFTHOMPSONSAMPLING
C.3.  Details of Equation(1)
Using Lemma 6 fory=μ
## 2
+ ∆/2, and∆
## ′
= ∆/2, we can bound the expected number of plays of
the second arm as:
## E[k
## 2
## (T)]=L+E
## 
## 
## T−1
## ∑
j=j
## 0
## Y
j
## 
## 
## ≤L+
## T−1
## ∑
j=0
## E
## [
min{E
## [
## X(j,s(j),μ
## 2
## +
## ∆
## 2
## )
s(j)
## ]
## , T}
## ]
## +
## ∑
t
## Pr(
## E
## 2
(t))·T
## ≤L+
4 lnT
## ∆
## ′2
## +
4(lnT)/∆
## ′2
## −1
## ∑
j=0
μ
## 1
## ∆
## ′
e
−Dj
## +
## (
y
## D
lnR
## )
## 2
## 1−y
## +
4(lnT)/∆
## ′2
## −1
## ∑
j=
y
## D
lnR
## R
y
e
−Dj
## 1−y
## +
## 16
## T
## ·T+ 2
## =L+
4 lnT
## ∆
## ′2
## +
4(lnT)/∆
## ′2
## −1
## ∑
j=0
μ
## 1
## ∆
## ′
e
−Dj
## +
y
## D
lnR·
## 2
## (1−y)
## +
4 lnT/∆
## ′2
## −
y
## D
lnR−1
## ∑
j=0
## 1
## 1−y
e
−Dj
## + 18
## ≤L+
4 lnT
## ∆
## ′2
## +
y
## D
lnR·
## 2
## ∆
## ′
## +
## T−1
## ∑
j=0
## (μ
## 1
## + 1)
## ∆
## ′
e
−Dj
## + 18
## (∗)
## ≤L+
4 lnT
## ∆
## ′2
## +
## D+ 1
## ∆
## ′
## D
## ·
## 2
## ∆
## ′
## +
## 2
## ∆
## ′
## 2
(min{D,1})
## + 18
## (∗∗)
## ≤L+
4 lnT
## ∆
## ′2
## +
## 2
## ∆
## ′2
## +
## 1
## ∆
## ′4
## +
## 4
## ∆
## ′3
## + 18
## =L+
16 lnT
## ∆
## 2
## +
## 8
## ∆
## 2
## +
## 16
## ∆
## 4
## +
## 32
## ∆
## 3
## + 18
## ≤
40 lnT
## ∆
## 2
## +
## 48
## ∆
## 4
## + 18.
The step marked(∗)is obtained using following derivations.
ylnR=yln
μ
## 1
## (1−y)
y(1−μ
## 1
## )
## =yln
μ
## 1
y
## +yln
## (1−y)
## (1−μ
## 1
## )
## ≤μ
## 1
## +
y
## 1−y
(D−yln
y
μ
## 1
## )≤1+
y
## 1−y
(D+μ
## 1
## )≤
## D+ 1
## ∆
## ′
## .
And, sinceD≥0(Gibbs’ inequality),
## ∑
j≥0
e
−Dj
## =
## 1
## 1−e
## −D
## ≤max{
## 2
## D
## ,
e
e−1
## }≤
## 2
min{D,1}
## .
And,(∗∗)uses Pinsker’s inequality to obtainD≥2∆
## ′
## 2
## .
C.4.  Proof of Lemma7
ProofThe proof of this lemma follows on the similar lines as the proof of Lemma5 in Appendix
C.1for the two arms case.  We will prove the second statement, the first statement will follow as a
corollary.
## 39.19

## AGRAWALGOYAL
To prove the second statement of this lemma, we are required to lower bound the probability of
Pr(E(t)|s(j) =s)for allt,j,s≤j, by1−
## 4(N−1)
## T
## 2
, wheres(j)denotes the number of successes
in firstjplays of the first arm. Recall that eventE(t)holds if the following is true:
{∀i∈C(t),θ
i
## (t)∈[μ
i
## −
## ∆
i
## 2
## ,μ
i
## +
## ∆
i
## 2
## ]}
Let us defineE
## +
i
(t)as the event{θ
i
## (t)≤μ
i
## +
## ∆
i
## 2
ori /∈C(t)}, andE
## −
i
(t)as the event{θ
i
## (t)≥
μ
i
## −
## ∆
i
## 2
ori /∈C(t)}. Then, we can boundPr(E(t)|s(j))as
Pr(E(t)|s(j))≤
## N
## ∑
i=2
## Pr(
## E
## +
i
## (t)|s(j)) + Pr(
## E
## −
i
## (t)|s(j)).
Now, observe that
## Pr(
## E
## +
i
## (t)|s(j)) = Pr(θ
i
(t)> μ
i
## +
## ∆
i
## 2
## ,k
i
(t)≥L
i
## |s(j)),
wherek
i
(t)is the number of plays of armiuntil timet−1.
As in the case of two arms, defineA
i
(t)as the event
## A
i
## (t) :
## S
i
## (t)
k
i
## (t)
## ≤μ
i
## +
## ∆
## 4
## ,
whereS
i
## (t),k
i
(t)denote the number of successes and number of plays respectively of thei
th
arm
until timet−1.
We will upper bound the probability ofPr(
## E
## +
i
(t)|s(j))for allt,j,i6= 1,using,
Pr(E
## +
i
## (t)|s(j))   =   Pr(θ
i
(t)> μ
i
## +
## ∆
i
## 2
## ,k
i
(t)≥L
i
## |s(j))
≤Pr(A
i
## (t),k
i
(t)≥L
i
## |s(j)) + Pr(θ
i
(t)> μ
i
## +
## ∆
i
## 2
## ,k
i
(t)≥L
i
## ,A
i
## (t)|s(j))
## (16)
For clarity of exposition, similar to the two arms case, for everyi= 1,...,Nwe define vari-
ables{Z
i,m
}, andZ
i,M
## .Z
i,m
denote the output of them
th
play of thei
th
arm. And,
## Z
i,M
## =
## 1
## M
## M
## ∑
m=1
## Z
i,m
Note that for alli,m,Z
i,m
is Bernoulli variable with meanμ
i
, and allZ
i,m
,i=  1,...,N,m=
1,...,Tare independent of each other.
Now,  instead  of  bounding  the  first  termPr(A
i
## (t),k
i
(t)≥L
i
|s(j)),  we  prove  a  bound  on
Pr(A(t),k
## 2
(t)≥L|Z
## 1,1
## ,...,Z
## 1,j
).   Note that the latter bound is stronger,  sinces(j)is simply
## ∑
j
m=1
## Z
## 1,m
## .
## 39.20

## ANALYSIS OFTHOMPSONSAMPLING
Now, for allt,i6= 1,
## Pr(
## A
i
## (t),k
i
(t)≥L
i
## |Z
## 1,1
## ,...,Z
## 1,j
## )   =
## ∑
## T
## `=L
## Pr(
## Z
i,k
i
## (t)
> μ
i
## +
## ∆
i
## 4
## ,k
i
(t) =`|Z
## 1,1
## ,...,Z
## 1,j
## )
## =
## ∑
## T
## `=L
Pr(Z
i,`
> μ
i
## +
## ∆
i
## 4
## ,k
i
(t) =`|Z
## 1,1
## ,...,Z
## 1,j
## )
## ≤
## ∑
## T
## `=L
Pr(Z
i,`
> μ
i
## +
## ∆
i
## 4
## |Z
## 1,1
## ,...,Z
## 1,j
## )
## =
## ∑
## T
## `=L
Pr(Z
i,`
> μ
i
## +
## ∆
i
## 4
## )
## ≤
## ∑
## T
## `=L
e
## −2`∆
## 2
i
## /16
## ≤
## 1
## T
## 2
The third last equality holds because for alli,i
## ′
## ,m,m
## ′
## ,Z
i,m
andZ
i
## ′
## ,m
## ′
are independent of each
other, which means
## Z
i,`
is independent ofZ
## 1,m
for allm= 1,...,j. The second last inequality is
by applying Chernoff bounds, sinceZ
i,`
is simply the average of`iid Bernoulli variables each with
meanμ
## 2
## .
We will derive the bound on second probability term in (16) in a similar manner.  As before, it
will be useful to defineW(`,z)as a random variable distributed as Beta(`z+ 1,`−`z+ 1). Note
that if at timet, the number of plays of armiisk
i
(t) =`, thenθ
i
(t)is distributed as Beta(`
## Z
i,`
## +
## 1,`−`Z
i,`
+ 1), i.e. same asW(`,Z
i,`
). Now, for the second probability term in (16),
## Pr(θ
i
(t)> μ
i
## +
## ∆
## 2
## ,A
i
## (t),k
i
(t)≥L
i
## |Z
## 1,1
## ,...,Z
## 1,j
## )
## =
## T
## ∑
## `=L
i
## Pr(θ
i
(t)> μ
i
## +
## ∆
i
## 2
## ,A
i
## (t),k
i
(t) =`|Z
## 1,1
## ,...,Z
## 1,j
## )
## ≤
## T
## ∑
## `=L
i
## Pr(θ
i
## (t)>
## S
i
## (t)
k
i
## (t)
## −
## ∆
i
## 4
## +
## ∆
i
## 2
## ,k
i
(t) =`|Z
## 1,1
## ,...,Z
## 1,j
## )
## =
## T
## ∑
## `=L
i
Pr(W(`,Z
i,`
## )>Z
i,`
## +
## ∆
i
## 4
## ,k
i
(t) =`|Z
## 1,1
## ,...,Z
## 1,j
## )
## ≤
## T
## ∑
## `=L
i
Pr(W(`,
## Z
i,`
## )>Z
i,`
## +
## ∆
i
## 4
## |Z
## 1,1
## ,...,Z
## 1,j
## )
## =
## T
## ∑
## `=L
i
Pr(W(`,Z
i,`
## )>Z
i,`
## +
## ∆
i
## 4
## )
## (using Fact 1)=
## T
## ∑
## `=L
i
## E
## [
## F
## B
## `+1,
## Z
i,`
## +
## ∆
i
## 4
## (`Z
i,`
## )
## ]
## ≤
## T
## ∑
## `=L
i
## E
## [
## F
## B
## `,Z
i,`
## +
## ∆
i
## 4
## (`
## Z
i,`
## )
## ]
## ≤
## T
## ∑
## `=L
i
exp{−
## 2∆
## 2
i
## `
## 2
## /16
## `
## }
## 39.21

## AGRAWALGOYAL
≤Te
## −2L
i
## ∆
## 2
i
## /16
## =
## 1
## T
## 2
## .
Here, we used the observation that for alli,i
## ′
## ,m,m
## ′
## ,Z
i,m
andZ
i
## ′
## ,m
## ′
are independent of each other,
which means
## Z
i,`
andW(`,Z
i,`
)are independent ofZ
## 1,m
for allm=  1,...,j.   The third-last
inequality follows from the observation that
## F
## B
n+1,p
(r) = (1−p)F
## B
n,p
(r) +pF
## B
n,p
(r−1)≤(1−p)F
## B
n,p
(r) +pF
## B
n,p
(r) =F
## B
n,p
## (r).
And,  the  second-last  inequality  follows  from  Chernoff–Hoeffding  bounds  (refer  to  Fact3  and
Lemma9). Substituting above in Equation (16), we get
## Pr(
## E
## +
i
## (t)|s(j))≤
## 2
## T
## 2
Similarly, we can obtain
## Pr(
## E
## −
i
## (t)|s(j))≤
## 2
## T
## 2
Summing overi= 2,...,N, we get
Pr(E(t)|s(j))≤
## 4(N−1)
## T
## 2
which implies the second statement of the lemma. The first statement is a simple corollary of this.
C.5.  Proof of Lemma8
## Proof
## E
## [
## ∑
γ
j
## +1
## `=1
## V
## `,a
j
s(j)
## ]
## =E
## [
## ∑
## T
## `=1
## V
## `,a
j
·I(γ
j
## ≥`−1)s(j)
## ]
LetF
## `−1
denote the history until before the beginning of intervalI
j
(`)(i.e. the values ofθ
i
## (t)and
the outcomes of playing the arms until the time step before the first time step ofI
j
(`)).  Note that
the value of random variable I(γ
j
≥`−1)is completely determined byF
## `−1
## . Therefore,
## E
## [
## ∑
γ
j
## +1
## `=1
## V
## `,a
j
s(j)
## ]
## =E
## [
## ∑
## T
## `=1
## E
## [
## V
## `,a
j
·I(γ
j
## ≥`−1)
s(j),F
## `−1
## ]
s(j)
## ]
## =E
## [
## ∑
## T
## `=1
## E
## [
## V
## `,a
j
s(j),F
## `−1
## ]
·I(γ
j
## ≥`−1)s(j)
## ]
## .
Recall thatV
## `,a
j
is the number of contiguous stepstfor whichais the best arm in saturated set
C(t)and iid variablesθ
## 1
(t)have value smaller thanμ
a
## +
## ∆
a
## 2
.   Observe that givens(j) =sand
## F
## `−1
## ,V
## `,a
j
is the length of an interval which ends when the value of an iid Beta(s+ 1,j−s+ 1)
distributed variable exceedsμ
a
## +
## ∆
a
## 2
(i.e.,M(t)happens), or if an arm other thanabecomes the best
saturated arm, or if we reach timeT. Therefore, givens(j),F
## `−1
## ,V
## `,a
j
is stochastically dominated
## 39.22

## ANALYSIS OFTHOMPSONSAMPLING
bymin{X(j,s(j),μ
a
## +
## ∆
a
## 2
),T}, where recall thatX(j,s(j),y)was defined as the number of trials
until an independent sample from Beta(s+ 1,j−s+ 1)distribution exceedsy. That is, for alla,
## E
## [
## V
## `,a
j
s(j),F
## `−1
## ]
## ≤E
## [
min{X(j,s(j),μ
a
## +
## ∆
a
## 2
## ),T}
s(j),F
## `−1
## ]
## =E
## [
min{X(j,s(j),μ
a
## +
## ∆
a
## 2
),T}s(j)
## ]
## .
Substituting, we get,
## E
## [
## ∑
γ
j
## +1
## `=1
## V
## `,a
j
s(j)
## ]
## ≤E
## [
## ∑
## T
## `=1
## E
## [
min{X(j,s(j),μ
a
## +
## ∆
a
## 2
),T}s(j)
## ]
·I(γ
j
## ≥`−1)s(j)
## ]
## =E
## [
min{X(j,s(j),μ
a
## +
## ∆
a
## 2
),T}s(j)
## ]
## ·E
## [
## ∑
## T
## `=1
## I(γ
j
## ≥`−1)s(j)
## ]
## =E
## [
min{X(j,s(j),μ
a
## +
## ∆
a
## 2
## ),T}
s(j)
## ]
·E[γ
j
## + 1s(j)].
This immediately implies,
## E
## [
## ∑
## N
a=2
## ∆
a
## E
## [
## ∑
γ
j
## +1
## `=1
## V
## `,a
j
s(j)
## ]]
## ≤E
## [
## ∑
## N
a=2
## ∆
a
## E
## [
min{X(j,s(j),μ
a
## +
## ∆
a
## 2
),T}s(j)
## ]
·E[γ
j
## + 1s(j)]
## ]
Appendix D.  Proof of Theorem2: details
We continue the proof from the main body of the paper.
By (6), regret due to playing saturated arms is bounded by
## ∑
## T−1
j=0
## E[R
s
## (I
j
## )]≤
## ∑
## T−1
j=0
## E
## [
## ∑
γ
j
## +1
## `=1
## ∑
## N
a=2
## 3∆
a
## V
## `,a
j
## ]
## + 2E
## [
## ∑
t∈I
j
## I(
## E(t))
## ]
## .(17)
Using Lemma 8, the regret contributed by the first term in (17) is bounded by
## 3
## ∑
## T−1
j=0
E[E[γ
j
s(j)]
## ∑
a
## ∆
a
E[min{X(j,s(j),y
a
## ),T}
s(j)]] +
## ∑
## T−1
j=0
E[E[min{X(j,s(j),y
a
## ),T}
s(j)]].
Recall thatγ
j
denotes the number of occurrences of eventM(t)in intervalI
j
, i.e.  the number
of times in intervalI
j
## ,θ
## 1
(t)was greater thanμ
i
## +
## ∆
i
## 2
of all saturated armsi∈C(t), and yet the
first arm was not played.  The only reasons the first arm would not be played at a timetdespite of
θ
## 1
## (t)>max
i∈C(t)
μ
i
## +
## ∆
i
## 2
are that eitherE(t)was violated, i.e.  some saturated arm whoseθ
i
## (t)
was not close to its mean was played instead; or some unsaturated armuwith highestθ
u
## (t)was
played.  Therefore, the random variablesγ
j
satisfy
γ
j
## ≤
## ∑
t∈I
j
I(an unsaturated arm is played at timet) +
## ∑
t∈I
j
I(E(t)).
Using Lemma 7, and the fact that an unsaturated armucan be played at mostL
u
times before it
becomes saturated, we obtain that
## ∑
## T−1
j=0
## E[γ
j
|s(j)]≤E[
## ∑
## T
t=1
I(an unsaturated arm is played at timet)|s(j)] +
## ∑
## T−1
j=0
## E[
## ∑
t∈I
j
## I(
## E(t))|s(j)]
## ≤
## ∑
u
## L
u
## +
## ∑
## T−1
j=0
## ∑
## T
t=1
## Pr(
## E(t)|s(j))
## ≤
## ∑
u
## L
u
## + 4(N−1).(18)
## 39.23

## AGRAWALGOYAL
Note that
## ∑
## T−1
j=0
## E[γ
j
|s(j)]is a r.v. (because of randoms(j)), and the above bound applies for
all instantiations of this r.v.
## Lety
a
## =μ
a
## +
## ∆
a
## 2
## . Then,
## E
## [
## ∑
## T−1
j=0
## E[γ
j
s(j)]
## ∑
a
## ∆
a
E[X(j,s(j),y
a
## )s(j)]
## ]
## ≤E
## [(
## ∑
## T−1
j=0
## E[γ
j
s(j)]
## )
## (max
j
## ∑
a
## ∆
a
E[X(j,s(j),y
a
## )s(j)])
## ]
## ≤(
## ∑
u
## L
u
## + 4(N−1))
## ∑
a
## ∆
a
## E[max
j
E[X(j,s(j),y
a
## )
s(j)]]
## ≤(
## ∑
u
## L
u
## + 4(N−1))
## ∑
a
## ∆
a
## E
## [
## ∆
a
## F
j
## ∗
a
## +1,y
a
## (s(j
## ∗
a
## ))
·I(s(j
## ∗
a
## )≤by
a
j
## ∗
a
c) +
## ∆
a
## F
j
## ∗
a
## +1,y
a
## (s(j
## ∗
a
## ))
·I(s(j
## ∗
a
## )≥dy
a
j
## ∗
a
e)
## ]
## ,
## (19)
where
j
## ∗
a
= argmax
j∈{0,...,T−1}
E[X(j,s(j),y
a
## )
s(j)] = argmax
j∈{0,...,T−1}
## 1
## F
j+1,y
a
## (s(j))
## .
Note thatj
## ∗
a
is a random variable, which is completely determined by the instantiation of random
sequences(1),s(2),....
For the first term in Equation (19),
## E
## [
## 1
## F
j
## ∗
a
## +1,y
a
## (s(j
## ∗
a
## ))
·I(s(j
## ∗
a
## )≤by
a
j
## ∗
a
c)
## ]
## ≤
## ∑
j
## E
## [
## 1
## F
j+1,y
a
## (s(j))
·I(s(j)≤by
a
jc)
## ]
## =
## ∑
j
by
a
jc
## ∑
s=0
f
j,μ
## 1
## (s)
## F
j+1,y
a
## (s)
## ≤
## ∑
j
μ
## 1
## ∆
## ′
a
e
## −D
a
j
## ≤
## 16
## ∆
## 3
a
## ,(20)
where∆
## ′
a
## =μ
## 1
## −y
a
## =  ∆
a
## /2,D
a
is the KL-divergence between Bernoulli distributions with
parametersμ
## 1
andy
a
.  The penultimate inequality follows using (13) in the proof of Lemma6 in
AppendixC.2, with∆
## ′
## = ∆
## ′
a
, andD=D
a
. The last inequality uses the geometric series sum (note
thatD
a
≥0by Gibbs’ inequality).
## ∑
j
e
## −D
a
j
## ≤
## 1
## 1−e
## −D
a
## ≤max{
## 2
## D
a
## ,
e
e−1
## }≤
## 2
min{D
a
## ,1}
## ≤
## 2
## ∆
## ′
a
## 2
## =
## 8
## ∆
## 2
a
## .
And, for the second term, using the fact thatF
j+1,y
(s)≥(1−y)F
j,y
(s), and that fors≥ dyje,
## F
j,y
(s)≥1/2(Fact 2),
## E
## [
## 1
## F
j
## ∗
a
## +1,y
a
## (s(j
## ∗
a
## ))
·I(s(j
## ∗
a
## )≥dy
a
j
## ∗
a
e)
## ]
## ≤
## 2
## 1−y
a
## ≤
## 4
## ∆
a
## .(21)
Substituting the bound from Equation (20) and (21) in Equation (19),
## ∑
## T−1
j=0
E[E[γ
j
## |s(j)]
## ∑
a
## 3∆
a
E[X(j,s(j),y
a
## )|s(j)]]≤(
## ∑
u
## L
u
## + 4(N−1))
## ∑
a
## (
## 48
## ∆
## 2
a
## + 12).
## (22)
## 39.24

## ANALYSIS OFTHOMPSONSAMPLING
Also, using Lemma6 while substitutingywithy
a
## =μ
a
## +
## ∆
a
## 2
and∆
## ′
withμ
## 1
## −y
a
## =
## ∆
a
## 2
## ,
## T−1
## ∑
j=0
## N
## ∑
a=2
## (3∆
a
## )E
## [
## E
## [
min{X(j,s(j),μ
a
## +
## ∆
a
## 2
),T}s(j)
## ]]
## ≤
## ∑
a
## (3∆
a
## )
16(lnT)
## ∆
a
## 2
## −1
## ∑
j=0
## (
## 1 +
## 2
## 1−y
a
## )
## +
## T
## ∑
j≥
16(lnT)
## ∆
a
## 2
## (3∆
a
## )
## 16
## T
## ≤
## ∑
a
48 lnT
## ∆
a
## +
## 192
## ∆
## 2
a
## + 48∆
a
## .(23)
Substituting bounds from (22) and (23) in the first term of Equation (17),
## T−1
## ∑
j=0
## E
## 
## 
γ
j
## +1
## ∑
## `=1
## ∑
a
## V
## `,a
j
## 3∆
a
## 
## 
## ≤(
## ∑
u
## L
u
## + 4(N−1))
## ∑
a
## (
## 48
## ∆
## 2
a
## + 12) +
## ∑
a
## (
48 lnT
## ∆
a
## +
## 192
## ∆
## 2
a
## + 48∆
a
## )
≤1152(lnT)(
## ∑
i
## 1
## ∆
## 2
i
## )
## 2
+ 288(lnT)
## ∑
i
## 1
## ∆
## 2
i
+ 48(lnT)
## ∑
a
## 1
## ∆
a
## + 192N
## ∑
a
## 1
## ∆
## 2
a
## + 96(N−1).
Now, using the result thatPr(E(t))≤4(N−1)/T
## 2
(by Lemma 7) with Equation (17), we can
bound the total regret due to playing saturated arms as
## E[R
s
## (T)]   =
## ∑
j
## E[R
s
## (I
j
## )]
## =
## ∑
j
## E
## 
## 
γ
j
## +1
## ∑
## `=1
## ∑
a
## V
## `,a
j
## 3∆
a
## 
## 
## + 2T·
## ∑
t
## Pr(
## E(t))
≤1152(lnT)(
## ∑
i
## 1
## ∆
## 2
i
## )
## 2
+ 288(lnT)
## ∑
i
## 1
## ∆
## 2
i
+48(lnT)
## ∑
a
## 1
## ∆
a
## + 192N
## ∑
a
## 1
## ∆
## 2
a
## + 96(N−1) + 8(N−1).
Since an unsaturated armubecomes saturated afterL
u
plays, regret due to unsaturated arms is at
most
## E[R
u
## (T)]≤
## N
## ∑
u=2
## L
u
## ∆
u
= 24(lnT)
## (
## N
## ∑
u=2
## 1
## ∆
u
## )
## .
Summing the regret due to saturated and unsaturated arms, we obtain the result of Theorem2.
## 39.25

## AGRAWALGOYAL
The proof for the alternate bound in Remark3 will essentially follow the same lines except that
instead of dividing the intervalI
j
(`)into subdivisionsV
## `,a
j
, we will simply bound the regret due to
saturated arms by number of plays times∆
max
. That is, we will use the bound,
## E[R(I
j
## )]≤E[
γ
j
## +1
## ∑
## `=1
## |I
j
## (`)|·∆
max
## ]
To boundE[
## ∑
γ
j
## +1
## `=1
## |I
j
(`)|], we follow the proof for boundingE[
## ∑
γ
j
## +1
## `=1
## V
## `, ̄a
j
]for ̄a= arg max
i6=1
μ
i
## ,
i.e., replacingμ
a
withμ
## ̄a
= max
i6=1
μ
## 1
, and∆
a
with∆
min
.  In a manner similar to Lemma8, we
can obtain
## E[
γ
j
## +1
## ∑
## `=1
## |I
j
(`)|]≤E[(γ
j
+ 1) min{X(j,s(j),μ
## M
## +
## ∆
min
## 2
## ),T}] +E[
## ∑
t∈I
j
T·I(E(t))]
And, consequently, using Equation (19), and Equation (20)–(23), and Lemma7, we can obtain
## ∑
j
## E[
γ
j
## +1
## ∑
## `=1
## |I
j
## (`)|]≤O((
## ∑
u
## L
u
## )
## 1
## ∆
## 3
min
## ) =O(
## 1
## ∆
## 3
min
## (
## N
## ∑
a=2
## 1
## ∆
## 2
a
## )
lnT),
giving a regret bound ofO(
## ∆
max
## ∆
## 3
min
## (
## ∑
## N
a=2
## 1
## ∆
## 2
a
## )
lnT).
## 39.26