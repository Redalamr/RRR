

Thompson Sampling for Contextual Bandits with Linear Payoffs
## Shipra Agrawalshipra@microsoft.com
## Microsoft Research India
## Navin Goyalnavingo@microsoft.com
## Microsoft Research India
## Abstract
Thompson   Sampling   is   one   of   the   old-
est  heuristics  for  multi-armed  bandit  prob-
lems.    It  is  a  randomized  algorithm  based
on  Bayesian  ideas,  and  has  recently  gener-
ated  significant  interest  after  several  stud-
ies demonstrated it to have better empirical
performance  compared  to  the  state-of-the-
art  methods.   However,  many  questions  re-
garding its theoretical performance remained
open.In  this  paper,  we  design  and  an-
alyze  a  generalization  of  Thompson  Sam-
pling algorithm for the stochastic contextual
multi-armed bandit problem with linear pay-
off functions, when the contexts are provided
by an adaptive adversary.  This is among the
most important and widely studied version of
the contextual bandits problem.  We prove a
high probability regret bound of
## ̃
## O(
d
## 2
## 
## √
## T
## 1+
## )
in  timeTfor  any  0<  <1,  wheredis
the  dimension  of  each  context  vector  and
is  a  parameter  used  by  the  algorithm.   Our
results  provide  the  first  theoretical  guaran-
tees for the contextual version of Thompson
Sampling,  and are close to the lower bound
of Ω(d
## √
T) for this problem.  This essentially
solves a COLT open problem of Chapelle and
Li [COLT 2012].
Proceedings of the 30
th
International Conference on Ma-
chine Learning, Atlanta, Georgia, USA, 2013.  JMLR:
W&CP volume 28. Copyright 2013 by the author(s).
## 1. Introduction
Multi-armed  bandit  (MAB)  problems  model  the  ex-
ploration/exploitation  trade-off  inherent  in  many  se-
quential decision problems.  There are many versions
of multi-armed bandit problems; a particularly useful
version is the contextual multi-armed bandit problem.
In this problem, in each ofTrounds, a learner is pre-
sented with the choice of taking one out ofNactions,
referred  to  asNarms.   Before  making  the  choice  of
which arm to play, the learner seesd-dimensional fea-
ture  vectorsb
i
,  referred  to  as  “context”,  associated
with each armi.  The learner uses these feature vec-
tors along with the feature vectors and rewards of the
arms played by her in the past to make the choice of
the arm to play in the current round.  Over time, the
learner’s  aim  is  to  gather  enough  information  about
how  the  feature  vectors  and  rewards  relate  to  each
other,  so  that  she  can  predict,  with  some  certainty,
which  arm  is  likely  to  give  the  best  reward  by  look-
ing at the feature vectors.  The learner competes with
a class of predictors, in which each predictor takes in
the  feature  vectors  and  predicts  which  arm  will  give
the  best  reward.   If  the  learner  can  guarantee  to  do
nearly as well as the predictions of the best predictor
in hindsight (i.e., have low regret), then the learner is
said to successfully compete with that class.
In  the  contextual  bandits  setting  withlinear  payoff
functions,  the  learner  competes  with  the  class  of  all
“linear”  predictors  on  the  feature  vectors.   That  is,
a  predictor  is  defined  by  ad-dimensional  parameter
μ∈R
d
, and the predictor ranks the arms according to
b
## T
i
μ.  We consider stochastic contextual bandit prob-
lem under linear realizability assumption, that is, we
assume that there is an unknown underlying parame-
terμ∈R
d
such that the expected reward for each arm
i, given contextb
i
, isb
## T
i
μ.  Under this realizability as-
sumption, the linear predictor corresponding toμis in
fact the best predictor and the learner’s aim is to learn
this underlying parameter.  This realizability assump-
tion is standard in the existing literature on contextual

Thompson Sampling for Contextual Bandits with Linear Payoffs
multi-armed bandits, e.g.  (Auer, 2002; Filippi et al.,
2010; Chu et al., 2011; Abbasi-Yadkori et al., 2011).
Thompson Sampling (TS) is one of the earliest heuris-
tics for multi-armed bandit problems. The first version
of this Bayesian heuristic is around 80 years old, dating
to Thompson (1933).  Since then, it has been rediscov-
ered numerous times independently in the context of
reinforcement learning, e.g., in Wyatt (1997); Ortega
& Braun (2010); Strens (2000).  It is a member of the
family ofrandomized probability matchingalgorithms.
The basic idea is to assume a simple prior distribution
on the underlying parameters of the reward distribu-
tion of every arm, and at every time step, play an arm
according to its posterior probability of being the best
arm.  The general structure of TS for the contextual
bandits problem involves the following elements:
-  a set Θ of parameters  ̃μ;
-  a prior distributionP( ̃μ) on these parameters;
-  past observationsDconsisting of (contextb,  re-
wardr) for the past time steps;
-  a  likelihood  functionP(r|b, ̃μ),  which  gives  the
probability of reward given a contextband a pa-
rameter  ̃μ;
-  a  posterior  distributionP( ̃μ|D)∝P(D| ̃μ)P( ̃μ),
whereP(D| ̃μ) is the likelihood function.
In each round, TS plays an arm according to its pos-
terior  probability  of  having  the  best  parameter.   A
simple way to achieve this is to produce a sample of
parameter for each arm, using the posterior distribu-
tions, and play the arm that produces the best sam-
ple.   In  this  paper,  we  design  and  analyze  a  natural
generalization  of  Thompson  Sampling  (TS)  for  con-
textual bandits; this generalization fits the above gen-
eral structure, and uses Gaussian prior and Gaussian
likelihood function.  We emphasize that although TS
is  a  Bayesian  approach,  the  description  of  the  algo-
rithm and our analysis apply to the prior-free stochas-
tic  MAB  model,  and  our  regret  bounds  will  hold  ir-
respective of whether or not the actual reward distri-
bution matches the Gaussian likelihood function used
to  derive  this  Bayesian  heuristic.   Thus,  our  bounds
for TS algorithm are directly comparable to the UCB
family of algorithms which form a frequentist approach
to the same problem.  One could interpret the priors
used by TS as a way of capturing the current knowl-
edge about the arms.
Recently,  TS  has  attracted  considerable  attention.
Several  studies  (e.g.,  Granmo  (2010);  Scott  (2010);
Graepel  et  al.  (2010);  Chapelle  &  Li  (2011);  May  &
Leslie  (2011);  Kaufmann  et  al.  (2012))  have  empiri-
cally  demonstrated  the  efficacy  of  TS:  Scott  (2010)
provides  a  detailed  discussion  of  probability  match-
ing techniques in many general settings along with fa-
vorable empirical comparisons with other techniques.
Chapelle & Li (2011) demonstrate that for the basic
stochastic MAB problem, empirically TS achieves re-
gret comparable to the lower bound of Lai & Robbins
(1985); and in applications like display advertising and
news article recommendation modeled by the contex-
tual  bandits  problem,  it  is  competitive  to  or  better
than the other methods such as UCB. In their exper-
iments, TS is also more robust to delayed or batched
feedback than the other methods.  TS has been used
in an industrial-scale application for CTR prediction
of search ads on search engines (Graepel et al., 2010).
Kaufmann et al. (2012) do a thorough comparison of
TS  with  the  best  known  versions  of  UCB  and  show
that TS has the lowest regret in the long run.
However, the theoretical understanding of TS is lim-
ited.Granmo  (2010)  and  May  et  al.  (2011)  pro-
vided  weak  guarantees,  namely,  a  bound  ofo(T)  on
the  expected  regret  in  timeT.For  the  the  ba-
sic  (i.e.   without  contexts)  version  of  the  stochastic
MAB problem, some significant progress was made by
Agrawal & Goyal (2012), Kaufmann et al. (2012) and,
more recently, by Agrawal & Goyal (2013), who pro-
vided  optimal  regret  bounds  on  the  expected  regret.
But, many questions regarding theoretical analysis of
TS  remained  open,  including  high  probability  regret
bounds, and regret bounds for the more general con-
textual bandits setting.  In particular, the contextual
MAB problem does not seem easily amenable to the
techniques used so far for analyzing TS for the basic
MAB  problem.   In  Section  3.1,  we  describe  some  of
these challenges.  Some of these questions and difficul-
ties  were  also  formally  raised  as  a  COLT  2012  open
problem (Chapelle & Li, 2012).
In  this  paper,  we  use  novel  martingale-based  analy-
sis techniques to demonstrate that TS (i.e., our Gaus-
sian  prior  based  generalization  of  TS  for  contextual
bandits)  achieves  high  probability,  near-optimal  re-
gret bounds for stochastic contextual bandits with lin-
ear payoff functions.  To our knowledge, ours are the
first non-trivial regret bounds for TS for the contex-
tual bandits problem.  Additionally, our results are the
first high probability regret bounds for TS, even in the
case of basic MAB problem.  This essentially solves the
COLT 2012 open problem by(Chapelle & Li, 2012) for
contextual bandits with linear payoffs.
Our version of Thompson Sampling algorithm for the
contextual MAB problem, described formally in Sec-
tion 2.2, uses Gaussian prior and Gaussian likelihood
functions.  Our techniques can be extended to the use
of  other  prior  distributions,  satisfying  certain  condi-

Thompson Sampling for Contextual Bandits with Linear Payoffs
tions, as discussed in Section 4.
- Problem setting and algorithm
description
2.1. Problem setting
There  areNarms.   At  timet=  1,2,...,  a  context
vectorb
i
(t)∈R
d
, is revealed for every armi.    These
context vectors are chosen by an adversary in an adap-
tive manner after observing the arms played and their
rewards up to timet−1, i.e.  historyH
t−1
## ,
## H
t−1
## ={a(τ),r
a(τ)
## (τ),b
i
(τ),i= 1,...,N,τ=
## 1,...,t−1},
wherea(τ) denotes the arm played at timeτ.  Given
b
i
(t), the reward for armiat timetis generated from
an (unknown) distribution with meanb
i
## (t)
## T
μ, where
μ∈R
d
is a fixed but unknown parameter.
## E
## [
r
i
## (t)
## {b
i
## (t)}
## N
i=1
## ,H
t−1
## ]
=E[r
i
## (t)b
i
## (t)] =b
i
## (t)
## T
μ.
An algorithm for thecontextual  bandit  problemneeds
to choose, at every timet, an arma(t) to play, using
historyH
t−1
and current contextsb
i
(t),i= 1,...,N.
## Leta
## ∗
(t) denote the optimal arm at timet, i.e.a
## ∗
## (t) =
arg max
i
b
i
## (t)
## T
μ.And  let  ∆
i
(t)  be  the  difference  be-
tween  the  mean  rewards  of  the  optimal  arm  and  of
armiat timet, i.e.,
## ∆
i
## (t) =b
a
## ∗
## (t)
## (t)
## T
μ−b
i
## (t)
## T
μ.
Then, the regret at timetis defined as
regret(t) = ∆
a(t)
## (t).
The objective is to minimize the total regretR(T) =
## ∑
## T
t=1
regret(t) in timeT.  The time horizonTis finite
but possibly unknown.
We assume thatη
i,t
## =r
i
## (t)−b
i
## (t)
## T
μis conditionally
R-sub-Gaussian for a constantR≥0, i.e.,
∀λ∈R,E[e
λη
i,t
## |{b
i
## (t)}
## N
i=1
## ,H
t−1
## ]≤exp
## (
λ
## 2
## R
## 2
## 2
## )
## .
This   assumption   is   satisfied   wheneverr
i
## (t)∈
## [b
i
## (t)
## T
μ−R,b
i
## (t)
## T
μ+R] (see Remark 1 in Appendix
A.1  of  Filippi  et  al.  (2010)).    We  will  also  assume
that||b
i
(t)|| ≤1,||μ|| ≤1,  and ∆
i
(t)≤1 for alli,t
(the norms, unless otherwise indicated, are`
## 2
## -norms).
These  assumptions  are  required  to  make  the  regret
bounds  scale-free,  and  are  standard  in  the  literature
on  this  problem.   If||μ|| ≤c,||b
i
## (t)|| ≤c,∆
i
## (t)≤c
instead,  then  our  regret  bounds  would  increase  by  a
factor ofc.
Remark  1.An  alternative  definition  of  regret  that
appears in the literature is
regret(t) =r
a
## ∗
## (t)
## (t)−r
a(t)
## (t).
We  can  obtain  the  same  regret  bounds  for  this  alter-
native definition of regret.  The details are provided in
the supplementary material in Appendix A.5.
2.2. Thompson Sampling algorithm
We  use  Gaussian  likelihood  function  and  Gaussian
prior to design our version of Thompson Sampling al-
gorithm.  More precisely, suppose that thelikelihood
of rewardr
i
(t) at timet, given contextb
i
(t) and pa-
rameterμ,  were given by the pdf of Gaussian distri-
butionN(b
i
## (t)
## T
μ,v
## 2
).  Here,v=R
## √
## 24
## 
dln(
## 1
δ
), with
∈(0,1) which parametrizes our algorithm.  Let
B(t) =I
d
## +
## ∑
t−1
τ=1
b
a(τ)
## (τ)b
a(τ)
## (τ)
## T
ˆμ(t) =B(t)
## −1
## (
## ∑
t−1
τ=1
b
a(τ)
## (τ)r
a(τ)
## (τ)
## )
## .
Then,   if  thepriorforμat  timetis  given  by
## N(ˆμ(t),v
## 2
## B(t)
## −1
),  it  is  easy  to  compute  theposte-
riordistribution at timet+ 1,
## Pr( ̃μ|r
i
(t))∝Pr(r
i
## (t)| ̃μ) Pr( ̃μ)
asN(ˆμ(t+ 1),v
## 2
## B(t+ 1)
## −1
) (details of this computa-
tion are in Appendix A.1). In our Thompson Sampling
algorithm, at every time stept, we will simply generate
a sample  ̃μ(t) from the distributionN(ˆμ(t),v
## 2
## B(t)
## −1
## ),
and play the armithat maximizesb
i
## (t)
## T
## ̃μ(t).
We emphasize that the Gaussian priors and the Gaus-
sian likelihood model for rewards are only used above
to design the Thompson Sampling algorithm for con-
textual bandits.  Our analysis of the algorithm allows
these models to be completely unrelated to theactual
reward  distribution.   The  assumptions  on  the  actual
reward distribution are only those mentioned in Sec-
tion 2.1, i.e., theR-sub-Gaussian assumption.
Algorithm  1Thompson  Sampling  for  Contextual
bandits
SetB=I
d
## ,ˆμ= 0
d
## ,f= 0
d
## .
for allt= 1,2,...,do
Sample  ̃μ(t) from distributionN(ˆμ,v
## 2
## B
## −1
## ).
Play arma(t) := arg max
i
b
i
## (t)
## T
̃μ(t), and observe
rewardr
t
## .
UpdateB=B+b
a(t)
## (t)b
a(t)
## (t)
## T
## ,f=f+
b
a(t)
## (t)r
t
, ˆμ=B
## −1
f.
end for
Every   steptof   Algorithm   1   consists   of   gener-
ating  ad-dimensional  sample    ̃μ(t)  from  a  multi-
variate Gaussian distribution, and solving the problem
arg max
i
b
i
## (t)
## T
̃μ(t).  Therefore, even if the number of
armsNis  large  (or  infinite),  the  above  algorithm  is
efficient as long as the problem arg max
i
b
i
## (t)
## T
̃μ(t) is

Thompson Sampling for Contextual Bandits with Linear Payoffs
efficiently solvable.  This is the case, for example, when
the set of arms at timetis given by ad-dimensional
convex set (every vector in the convex set is a context
vector, and thus corresponds to an arm).
## 2.3. Our Results
Theorem   1.For   the   stochastic   contextual   ban-
dit  problem  with  linear  payoff  functions,  with  prob-
ability1−δ,    the   total   regret   in   timeTfor
Thompson   Sampling   (Algorithm   1)   is   bounded   by
## O
## (
d
## 2
## 
## √
## T
## 1+
## (
ln(Td) ln
## 1
δ
## )
## )
,  for  any0<  <1,0<
δ <1.  Here,is  a  parameter  used  by  the  Thompson
Sampling algorithm.
Remark 2.The parametercan be chosen to be any
constant  in(0,1).   IfTis  known,  one  could  choose
## =
## 1
lnT
, to get
## ̃
## O(d
## 2
## √
T)regret bound.
Remark 3.Our regret bound in Theorem 1 does not
depend  onN,  and  is  applicable  to  the  case  of  infi-
nite arms, with only notational changes required in the
analysis.
In  the  main  body  of  this  paper,  we  will  discuss  the
proof  of  the  above  result.   Below,  we  state  two  ad-
ditional results; their proofs require small changes to
the proof of Theorem 1 and are provided in the sup-
plementary material.
The first result is for the setting where each of theN
arms is associated with a differentd-dimensional pa-
rameterμ
i
## ∈R
d
, so that the mean reward for armi
at  timetisb
i
## (t)
## T
μ
i
.   This  setting  is  a  direct  gener-
alization of the basic MAB problem tod-dimensions.
Thompson  Sampling  for  this  setting  will  maintain  a
separate  posterior  distribution  for  each  armiwhich
would be updated only at the time instances wheniis
played.  And, at every time stept, instead of a single
sample   ̃μ(t),Nindependent  samples  will  have  to  be
generated:   ̃μ
i
(t) for each armi.  We prove the follow-
ing regret bound for this setting.
Theorem  2.For  the  setting  withNdifferent  pa-
rameters,   with   probability1−δ,   the   total   regret
in  timeTfor  Thompson  Sampling  is  bounded  by
## O
## (
d
## √
## NT
## 1+
lnN
## 
## (
lnTln
## 1
δ
## )
## )
,  for  any0<  <1,
0< δ <1.
The details of the algorithm forN-parameter setting
and the proof of Theorem 2 appear in the supplemen-
tary material in Appendix C.
Note that unlike Theorem 1, the regret bound in The-
orem  2  has  a  dependence  onN,  which  is  expected
because  Theorem  2  deals  with  a  setting  where  there
areNdifferent  parameters  to  learn.    However,  the
bound  in  Theorem  2  has  a  better  dependence  ond.
This  improvement  results  from  the  independence  of
θ
i
## (t)  =b
i
## (t)
## T
## ̃μ
i
(t)  in  the  algorithm  for  this  setting.
On the other hand in Algorithm 1, used for the single
parameter setting of Theorem 1, a single   ̃μ(t) is gen-
erated, and soθ
i
## (t) =b
i
## (t)
## T
̃μ(t) are not independent.
This  motivates  us  to  consider  a  modification  of  Al-
gorithm  1  for  the  single  parameter  setting,  in  which
theθ
i
(t)’s  are  independently  generated,  each  with
marginal  distributionb
i
## (t)
## T
̃μ(t).   The  arm  with  the
highest value ofθ
i
(t) is played at timet. Although, this
modified  algorithm  could  be  inefficient  compared  to
Algorithm 1 ifNis large (say exponential) compared
tod, the better dependence ondin regret bounds could
be useful ifdis large.
Theorem 3.For the modified algorithm in single pa-
rameter setting, with probability1−δ, the total regret
in  timeTis  bounded  byO
## (
d
## √
## T
## 1+
lnN
## 
## (
lnTln
## 1
δ
## )
## )
## ,
for any0<  <1,0< δ <1.
The  details  of  the  modified  algorithm  and  the  proof
of  the  above  theorem  appears  in  the  supplementary
material in Appendix B.
## 2.4. Related Work
The contextual bandit problem with linear payoffs is
a  widely  studied  problem  in  statistics  and  machine
learning often under different names as mentioned by
Chu  et  al.  (2011):  bandit  problems  with  co-variates
(Woodroofe, 1979; Sarkar, 1991), associative reinforce-
ment  learning  (Kaelbling,  1994),  associative  bandit
problems (Auer, 2002; Strehl et al., 2006), bandit prob-
lems with expert advice (Auer et al., 2002), and linear
bandits (Dani et al., 2008; Abbasi-Yadkori et al., 2011;
Bubeck et al., 2012).  The namecontextual banditswas
coined in Langford & Zhang (2007).
A lower bound of Ω(d
## √
T) for this problem was given
by Dani et al. (2008), when the number of arms is al-
lowed  to  be  infinite.   In  particular,  they  prove  their
lower bound using an example where the set of arms
correspond  to  all  vectors  in  the  intersection  of  ad-
dimensional  sphere  and  a  cube.   They  also  provide
an upper bound of
## ̃
## O(d
## √
T), although their setting is
slightly restrictive in the sense that the context vector
for every arm is fixed in advanced and is not allowed to
change with time.  Abbasi-Yadkori et al. (2011) ana-
lyze a UCB-style algorithm and provide a regret upper
bound ofO(dlog (T)
## √
## T+
## √
dTlog (T/δ)). Apart from
the dependence on, our bounds are essentially away
by a factor ofdfrom these bounds.
For  finiteN,  Chu  et  al.  (2011)  show  a  lower  bound

Thompson Sampling for Contextual Bandits with Linear Payoffs
of  Ω(
## √
Td)  ford
## 2
≤T.Auer  (2002)  and  Chu
et  al.  (2011)  analyze  SupLinUCB,  a  complicated  al-
gorithm  using  UCB  as  a  subroutine,  for  this  prob-
lem.    Chu  et  al.  (2011)  achieve  a  regret  bound  of
## O(
## √
## Tdln
## 3
(NTln(T)/δ)) with probability at least 1−
δ(Auer  (2002)  proves  similar  results).   This  regret
bound  is  not  applicable  to  the  case  of  infinite  arms,
and assumes that context vectors are generated by an
obliviousadversary. Also, this regret bound would give
## O(d
## 2
## √
T) regret ifNis exponential ind.  The state-
of-the-art  bounds  for  linear  bandits  problem  in  case
of  finiteNare  given  by  Bubeck  et  al.  (2012).   They
provide  an  algorithm  based  on  exponential  weights,
with regret of order
## √
dTlogNfor any finite set ofN
actions.  However, the exponential weights based algo-
rithms  are  not  efficient  ifNis  large  (sampling  com-
plexity  ofO(N)  in  every  step).   Also,  their  setting
is  slightly  different  from  ours.   The  set  of  arms  and
the associatedb
i
vectors arenon-adaptiveand fixed in
advance.  And, they consider a non-stochastic (adver-
sarial) bandit setting where the reward at timetfor
armiisb
## T
i
μ
t
withμ
t
chosen by an adversary.
Very recent work Russo & Roy (2013) provides near-
optimal  bounds  onBayesian  regretin  many  general
settings.  This result is incomparable to ours because
of the different notion of regret used.
While the regret bounds provided in this paper do not
match  or  better  the  best  available  regret  bounds  for
the  extensively  studied  problem  of  linear  contextual
bandits, our results demonstrate that the natural and
efficient heuristic of Thompson Sampling can achieve
theoretical bounds that are close to the best bounds.
The main contribution of this paper is to provide new
tools for analysis of Thompson Sampling algorithm for
contextual bandits,  which despite being popular and
empirically attractive, has eluded theoretical analysis.
We believe the techniques used in this paper will pro-
vide useful insights into the workings of this Bayesian
algorithm, and may be useful for further improvements
and extensions.
- Regret Analysis:  Proof of Theorem 1
3.1. Challenges and proof outline
The  contextual  version  of  the  multi-armed  bandit
problem presents new challenges for the analysis of TS
algorithm, and the techniques used so far for analyz-
ing the basic multi-armed bandit problem by Agrawal
& Goyal (2012); Kaufmann et al. (2012) do not seem
directly applicable.  Let us describe some of these dif-
ficulties and our novel ideas to resolve them.
In  the  basic  MAB  problem  there  areNarms,  with
mean  rewardμ
i
∈Rfor  armi,  and  the  regret  for
playing a suboptimal armiisμ
a
## ∗
## −μ
i
, wherea
## ∗
is the
arm with the highest mean. Let us compare this to a 1-
dimensional contextual MAB problem, where armiis
associated with a parameterμ
i
∈R, but in addition, at
every timet, it is associated with a contextb
i
(t)∈R,
so that mean reward isb
i
## (t)μ
i
.  The best arma
## ∗
(t) at
timetis the arm with the highest mean at timet, and
the regret for playing armiisb
a
## ∗
## (t)
## (t)μ
a
## ∗
## (t)
## −b
i
## (t)μ
i
## .
In  general,  the  basis  of  regret  analysis  for  stochastic
MAB is to prove that the variances of empirical esti-
mates for all arms decrease fast enough, so that the re-
gret incurred until the variances become small enough,
is small.  In the basic MAB, the variance of the em-
pirical  mean  is  inversely  proportional  to  the  number
of playsk
i
(t) of armiat timet.  Thus, every time the
suboptimal armiis played, we know that even though
a regret ofμ
i
## ∗
## −μ
i
≤1 is incurred, there is also an im-
provement of exactly 1 in the number of plays of that
arm,  and  hence,  corresponding  decrease  in  the  vari-
ance.  The techniques for analyzing basic MAB rely on
this observation to precisely quantify the exploration-
exploitation tradeoff.  On the other hand, the variance
of the empirical mean for the contextual case is given
by inverse ofB
i
## (t) =
## ∑
t
τ=1:a(τ)=i
b
i
## (τ)
## 2
.  When a sub-
optimal  armiis  played,  ifb
i
(t)  is  small,  the  regret
b
a
## ∗
## (t)
## (t)μ
a
## ∗
## (t)
## −b
i
## (t)μ
i
could be much higher than the
improvementb
i
## (t)
## 2
inB
i
## (t).
In  our  proof,  we  overcome  this  difficulty  by  dividing
the arms into two groups at any time:  saturated and
unsaturated  arms,  based  on  whether  the  standard
deviation  of  the  estimates  for  an  arm  is  smaller  or
larger  compared  to  the  standard  deviation  for  the
optimal  arm.    The  optimal  arm  is  included  in  the
group  of  unsaturated  arms.    We  show  that  for  the
unsaturated arms, the regret on playing the arm can
be  bounded  by  a  factor  of  the  standard  deviation,
which  improves  every  time  the  arm  is  played.   This
allows us to bound the total regret due to unsaturated
arms.  For the saturated arms,  standard deviation is
small,  or in other words,  the estimates of the means
constructed so far are quite accurate in the direction
of  the  current  contexts  of  these  arms,  so  that  the
algorithm  is  able  to  distinguish  between  them  and
the optimal arm.  We utilize this observation to show
that the probability of playing such arms at any step
is bounded by a function of the probability of playing
the unsaturated arms.
Below  is  a  more  technical  outline  of  the  proof  of
Theorem  1.   At  any  time  stept,  we  divide  the  arms
into two groups:

Thompson Sampling for Contextual Bandits with Linear Payoffs
•saturated  armsdefined  as  those  withg(T)s
t,i
## <
`(T)s
t,a
## ∗
## (t)
## ,
•unsaturated armsdefined as those withg(T)s
t,i
## ≥
`(T)s
t,a
## ∗
## (t)
## ,
wheres
t,i
## =
## √
b
i
## (t)
## T
## B(t)
## −1
b
i
(t)  andg(T),`(T)
(g(T)> `(T)) are constants (functions ofT,d,δ) de-
fined later.  Note thats
t,i
is the standard deviation of
the estimateb
i
## (t)
## T
ˆμ(t) andvs
t,i
is the standard devi-
ation of the random variableb
i
## (t)
## T
## ̃μ(t).
We  use  concentration  bounds  for   ̃μ(t)  and  ˆμ(t)  to
bound the regret at any timetbyg(T)(s
t,a
## ∗
## (t)
## +s
t,a(t)
## ).
Now, if an  unsaturated arm is played at timet, then
using the definition of  unsaturated arms, the regret is
at most
2g(T)
## 2
## `(T)
s
t,a(t)
.  This is useful because of the in-
equality
## ∑
t
s
t,a(t)
## =O(
## √
TdlnT) (derived along the
lines  of  Auer  (2002)),  which  allows  us  to  bound  the
total regret due to unsaturated arms.
For saturated arms, we prove that the probability of
playing a  saturated arm at any timetis withinpof the
probability of playing an unsaturated arm, wherep=
## 1
## 4e
## √
πT
## 
.  More precisely, we defineF
t−1
as the union
of historyH
t−1
and the contextsb
i
(t),i= 1,...,Nat
timet, and prove that for “most” (in a high probability
sense)F
t−1
## ,
Pr (a(t) is a saturated arm
## F
t−1
## )≤
## 1
p
·Pr (a(t) is an unsaturated arm
## F
t−1
## ) +
## 1
pT
## 2
## ,
We use these observations to establish that (X
t
## ;t≥0),
where
## X
t
## 'regret(t)−
g(T)
p
I(a(t) is unsaturated)s
t,a
## ∗
## (t)
## −
2g(T)
## 2
## `(T)
s
t,a(t)
## −
2g(T)
pT
## 2
## ,
is a super-martingale difference process adapted to fil-
trationF
t
.  Then, using the Azuma-Hoeffding inequal-
ity  for  super-martingales,  along  with  the  inequality
## ∑
t
s
t,a(t)
## =O(
## √
TdlnT),  we  will  obtain  the  desired
high probability regret bound.
3.2. Formal proof
For  quick  reference,  the  notations  introduced  below
also appear in a table of notations at the beginning of
the supplementary material.
Definition  1.For  alli,  defineθ
i
## (t)  =b
i
## (t)
## T
## ̃μ(t),
ands
t,i
## =
## √
b
i
## (t)
## T
## B(t)
## −1
b
i
(t).  By  definition  of ̃μ(t),
marginal  distribution  of  eachθ
i
(t)is  Gaussian  with
meanb
i
## (t)
## T
ˆμ(t)and  standard  deviationvs
t,i
## .   Also,
s
t,i
is the standard deviation of estimateb
i
## (t)
## T
## ˆμ(t).
Definition  2.Recall   that∆
i
## (t)   =b
a
## ∗
## (t)
## (t)
## T
μ−
b
i
## (t)
## T
μ, the difference between the mean reward of op-
timal arm and armiat timet.  .
Definition  3.Define`(T)  =R
## √
dln(T
## 3
) ln(
## 1
δ
## ) + 1,
v=R
## √
## 24
## 
dln(
## 1
δ
), andg(T) =
## √
4dln(Td)v+`(T).
Definition 4.DefineE
μ
(t)andE
θ
(t)as  the  events
thatb
i
## (t)
## T
## ˆμ(t)andθ
i
(t)are concentrated around their
respective  means.  More  precisely,  defineE
μ
(t)as  the
event that
## ∀i:|b
i
## (t)
## T
## ˆμ(t)−b
i
## (t)
## T
μ|≤`(T)s
t,i
## .
DefineE
θ
(t)as the event that
## ∀i:|θ
i
## (t)−b
i
## (t)
## T
## ˆμ(t)|≤
## √
4dln(Td)vs
t,i
## .
Definition 5.An armiis calledsaturatedat time
tifg(T)s
t,i
< `(T)s
t,a
## ∗
## (t)
,  andunsaturatedoth-
erwise.  LetC(t)denote  the  set  of   saturated  arms  at
timet.   Note  that  the  optimal  arm  is  always  unsatu-
rated  at  timet,  i.e.,a
## ∗
(t)/∈C(t).  An  arm  may  keep
shifting from   saturated to   unsaturated and vice-versa
over time.
Definition 6.Define  filtrationF
t−1
as  the  union  of
history until timet−1, and the contexts at timet, i.e.,
## F
t−1
## ={H
t−1
## ,b
i
(t),i= 1,...,N}.
By  definition,F
## 1
## ⊆ F
## 2
## ··· ⊆ F
## T−1
.  Observe  that  the
following quantities are determined by the historyH
t−1
and the contextsb
i
(t)at timet, and hence are included
inF
t−1
## ,
•ˆμ(t),B(t),
## •s
t,i
, for alli,
•the  identity  of  the  optimal  arma
## ∗
(t)and  the  set
of saturated armsC(t),
•whetherE
μ
(t)is true or not,
•thedistributionN(ˆμ(t),B(t)
## −1
## )of ̃μ(t),
and   hence   the   joint   distribution   ofθ
i
## (t)    =
b
i
## (t)
## T
̃μ(t),i= 1,...,N.
Lemma 1.For allt,0< δ <1,Pr(E
μ
## (t))≥1−
δ
## T
## 2
## .
And, for all possible filtrationsF
t−1
,Pr(E
θ
(t)|F
t−1
## )≥
## 1−
## 1
## T
## 2
## .
Proof.The  complete  proof  of  this  lemma  appears  in
Appendix A.3.  The probability bound forE
μ
(t) will
be  proven  using  a  concentration  inequality  given  by
Abbasi-Yadkori  et  al.  (2011),  stated  as  Lemma  7  in
Appendix  A.2.   TheR-sub-Gaussian  assumption  on
rewards will be utilized here.  The probability bound
forE
θ
(t) will be proven using a concentration inequal-
ity  for  Gaussian  random  variables  from  Abramowitz
& Stegun (1964) stated as Lemma 5 in Appendix A.2
## .
The  next  lemma  lower  bounds  the  probability  that
θ
a
## ∗
## (t)
## (t)   =b
a
## ∗
## (t)
## (t)
## T
̃μ(t)  for  the  optimal  arm  at
timetwill  exceed  its  mean  rewardb
a
## ∗
## (t)
## (t)
## T
μplus
`(T)s
t,a
## ∗
## (t)
## .

Thompson Sampling for Contextual Bandits with Linear Payoffs
Lemma 2.For any filtrationF
t−1
such thatE
μ
## (t)is
true,
## Pr
## (
θ
a
## ∗
## (t)
(t)> b
a
## ∗
## (t)
## (t)
## T
μ+`(T)s
t,a
## ∗
## (t)
## F
t−1
## )
## ≥
## 1
## 4e
## √
πT
## 
## .
Proof.The proof uses anti-concentration of Gaussian
random variableθ
a
## ∗
## (t)
## (t) =b
a
## ∗
## (t)
## (t)
## T
̃μ(t),  which has
meanb
a
## ∗
## (t)
## (t)
## T
ˆμ(t)  and  standard  deviationvs
t,a
## ∗
## (t)
## ,
provided by Lemma 5 in Appendix A.2, and the con-
centration  ofb
a
## ∗
## (t)
## (t)
## T
ˆμ(t)  aroundb
a
## ∗
## (t)
## (t)
## T
μpro-
vided  by  the  eventE
μ
(t).   The  details  of  the  proof
are in Appendix A.4.
The following lemma bounds the probability of playing
saturated arms in terms of the probability of playing
unsaturated arms.
Lemma 3.Given any filtrationF
t−1
such thatE
μ
## (t)
is true,
Pr (a(t)∈C(t)F
t−1
## )≤
## 1
p
Pr (a(t)/∈C(t)F
t−1
## ) +
## 1
pT
## 2
## ,
wherep=
## 1
## 4e
## √
πT
## 
## .
Proof.The algorithm chooses the arm with the high-
est value ofθ
i
## (t) =b
i
## (t)
## T
̃μ(t) to be played at timet.
Therefore, ifθ
a
## ∗
## (t)
(t) is greater thanθ
j
(t) for all satu-
rated arms, i.e.,θ
a
## ∗
## (t)
(t)> θ
j
(t),∀j∈C(t), then one
of  the  unsaturated  arms  (which  include  the  optimal
arm and other suboptimal unsaturated arms) must be
played.  Therefore,
Pr (a(t)/∈C(t)
## F
t−1
## )
≥Pr
## (
θ
a
## ∗
## (t)
(t)> θ
j
(t),∀j∈C(t)F
t−1
## )
## .(1)
By  definition,  for  all  saturated  arms,  i.e.for  all
j∈C(t),g(T)s
t,j
< `(T)s
t,a
## ∗
## (t)
.   Also,  if  both  the
eventsE
μ
(t)  andE
θ
(t)  are  true  then,  by  the  def-
initions  of  these  events,  for  allj∈C(t),θ
j
## (t)≤
b
j
## (t)
## T
μ+g(T)s
t,j
.    Therefore,  given  anF
t−1
such
thatE
μ
(t) is true, eitherE
θ
(t) is false, or else for all
j∈C(t),
θ
j
## (t)≤b
j
## (t)
## T
μ+g(T)s
t,j
## ≤b
a
## ∗
## (t)
## (t)
## T
μ+`(T)s
t,a
## ∗
## (t)
## .
Hence, for anyF
t−1
such thatE
μ
(t) is true,
## Pr
## (
θ
a
## ∗
## (t)
(t)> θ
j
(t),∀j∈C(t)
## F
t−1
## )
≥Pr
## (
θ
a
## ∗
## (t)
(t)> b
a
## ∗
## (t)
## (t)
## T
μ+`(T)s
t,a
## ∗
## (t)
## F
t−1
## )
−Pr
## (
## E
θ
## (t)
## F
t−1
## )
## ≥p−
## 1
## T
## 2
## .
The last inequality uses Lemma 2 and Lemma 1.  Sub-
stituting in Equation (1), this gives,
Pr (a(t)/∈C(t)F
t−1
## ) +
## 1
## T
## 2
## ≥p,
which implies
Pr (a(t)∈C(t)F
t−1
## )
Pr (a(t)/∈C(t)F
t−1
## ) +
## 1
## T
## 2
## ≤
## 1
p
## .
Definition  7.Recall  that  regret(t)was  defined  as,
regret(t) = ∆
a(t)
## (t) =b
a
## ∗
## (t)
## (t)
## T
μ−b
a(t)
## (t)
## T
μ.Define
regret
## ′
(t) =regret(t)·I(E
μ
## (t)).
Next, we establish a super-martingale process that will
form the basis of our proof of the high-probability re-
gret bound.
Definition 8.Let
## X
t
## =regret
## ′
## (t)−
g(T)
p
I(a(t)/∈C(t))s
t,a
## ∗
## (t)
## −
2g(T)
## 2
## `(T)
s
t,a(t)
## −
2g(T)
pT
## 2
## ,
## Y
t
## =
## ∑
t
w=1
## X
w
## ,
wherep=
## 1
## 4e
## √
πT
## 
## .
Lemma  4.(Y
t
;t=  0,...,T)is  a  super-martingale
process with respect to filtrationF
t
## .
Proof.See Definition 9 in Appendix A.2 for the defi-
nition of super-martingales.  We need to prove that for
allt∈[1,T], and anyF
t−1
## ,E[Y
t
## −Y
t−1
## |F
t−1
]≤0, i.e.
## E[regret
## ′
## (t)
## F
t−1
## ]≤
g(T)
p
Pr (a(t)/∈C(t)F
t−1
## )s
t,a
## ∗
## (t)
## +
2g(T)
## 2
## `(T)
## E
## [
s
t,a(t)
## F
t−1
## ]
## +
2g(T)
pT
## 2
## .
IfF
t−1
is such thatE
μ
(t) is not true, then regret
## ′
## (t) =
regret(t)·I(E
μ
(t)) = 0, and the above inequality holds
trivially.  So, we considerF
t−1
such thatE
μ
(t) holds.
We  observe  that  if  the  eventsE
μ
(t),E
θ
(t)  are  true,
then ∆
a(t)
(t)≤g(T)(s
t,a(t)
## +s
t,a
## ∗
## (t)
).  This is because
if an armiis played at timet,  then it must be true
thatθ
i
## (t)≥θ
a
## ∗
## (t)
(t).   And,  ifE
θ
(t)  andE
μ
(t)  are
true, then,
b
i
## (t)
## T
μ≥θ
i
(t)−g(T)s
t,i
## ≥θ
a
## ∗
## (t)
(t)−g(T)s
t,i
## ≥b
a
## ∗
## (t)
## (t)
## T
μ−g(T)s
t,a
## ∗
## (t)
−g(T)s
t,i
## .
Therefore,  given  a  filtrationF
t−1
such  thatE
μ
(t)  is
true, either ∆
a(t)
(t)≤g(T)(s
t,a(t)
## +s
t,a
## ∗
## (t)
) orE
θ
## (t)

Thompson Sampling for Contextual Bandits with Linear Payoffs
is false.  And, hence,
## E[regret
## ′
(t)F
t−1
## ]
## =E
## [
## ∆
a(t)
(t)F
t−1
## ]
## ≤E
## [
g(T)(s
t,a
## ∗
## (t)
## +s
t,a(t)
## )F
t−1
## ]
## + Pr
## (
## E
θ
## (t)
## )
=g(T)E
## [
s
t,a
## ∗
## (t)
I(a(t)∈C(t))F
t−1
## ]
+g(T)E
## [
s
t,a
## ∗
## (t)
I(a(t)/∈C(t))F
t−1
## ]
+g(T)E
## [
s
t,a(t)
## F
t−1
## ]
## + Pr
## (
## E
θ
## (t)
## )
≤g(T)s
t,a
## ∗
## (t)
Pr (a(t)∈C(t)F
t−1
## )
+g(T)E
## [(
g(T)
## `(T)
## )
s
t,a(t)
I(a(t)/∈C(t))
## F
t−1
## ]
+g(T)E
## [
s
t,a(t)
## F
t−1
## ]
## +
## 1
## T
## 2
≤g(T)s
t,a
## ∗
## (t)
## ·
## 1
p
Pr (a(t)/∈C(t)
## F
t−1
) +g(T)
## 1
pT
## 2
## +
## (
2g(T)
## 2
## `(T)
## )
## E
## [
s
t,a(t)
## F
t−1
## ]
## +
## 1
## T
## 2
≤g(T)s
t,a
## ∗
## (t)
## ·
## 1
p
Pr (a(t)/∈C(t)F
t−1
## )
## +
## (
2g(T)
## 2
## `(T)
## )
## E
## [
s
t,a(t)
## F
t−1
## ]
## +
2g(T)
pT
## 2
## .
In the first inequality we used that for alli, ∆
i
## (t)≤1.
The  second  inequality  used  the  definition  of  unsatu-
rated arms to applys
t,a
## ∗
## (t)
## ≤
g(T)
## `(T)
s
t,a(t)
, and Lemma
1 to apply Pr
## (
## E
θ
## (t)
## )
## ≤
## 1
## T
## 2
. The third inequality used
Lemma 3, and also the observation that 0≤s
t,a
## ∗
## (t)
## ≤
## ||b
a
## ∗
## (t)
## (t)||≤1.
Now, we are ready to prove Theorem 1.
Proof of Theorem 1We observe that the absolute
value of each of the four terms in the definition ofX
t
is bounded by
## 2
p
g(T)
## 2
## `(T)
, therefore the super-martingale
## Y
t
has bounded difference|Y
t
## −Y
t−1
## |≤
## 8
p
g(T)
## 2
## `(T)
, for all
t≥1. Thus, we can apply Azuma-Hoeffding inequality
(see Lemma 6 in Appendix A.2), to obtain that with
probability 1−
δ
## 2
## ,
## ∑
## T
t=1
regret
## ′
## (t)
## ≤
## ∑
## T
t=1
## (
g(T)
p
I(a(t)/∈C(t))s
t,a
## ∗
## (t)
## )
## +
2g(T)
pT
## +
2g(T)
## 2
## `(T)
## ∑
## T
t=1
s
t,a(t)
## +
## 8
p
g(T)
## 2
## `(T)
## √
2Tln(
## 2
δ
## )
## ≤
## ∑
## T
t=1
## (
g(T)
## 2
## `(T)
## 1
p
I(a(t)/∈C(t))s
t,a(t)
## )
## +
2g(T)
pT
## +
2g(T)
## 2
## `(T)
## ∑
## T
t=1
s
t,a(t)
## +
## 8
p
g(T)
## 2
## `(T)
## √
2Tln(
## 2
δ
## )
## ≤
g(T)
## 2
## `(T)
## 3
p
## ∑
## T
t=1
s
t,a(t)
## +
2g(T)
pT
## +
## 8
p
g(T)
## 2
## `(T)
## √
2Tln(
## 2
δ
## ).
The second inequality used the observation that if an
unsaturated  arm  is  played,  i.e.,a(t)/∈C(t),  then,
g(T)s
t,a(t)
≥`(T)s
t,a
## ∗
## (t)
## .
Now,  we  can  use
## ∑
## T
t=1
s
t,a(t)
## ≤5
## √
dTlnT,which
can  be  derived  along  the  lines  of  Lemma  3  of  Chu
et al. (2011) using Lemma 11 of Auer (2002) (see Ap-
pendix A.5 for details).  Also, recalling the definitions
ofp,`(T), andg(T) (see the Table of notations in the
beginning of the supplementary material), and substi-
tuting in above, we get
## ∑
## T
t=1
regret
## ′
(t) =O
## (
d
## 2
## 
## √
## T
## (1+)
ln(
## 1
δ
) ln(Td)
## )
## .
Also,  becauseE
μ
(t)  holds  for  alltwith  probability
at  least  1−
δ
## 2
(see  Lemma  1),  regret
## ′
(t)  =  regret(t)
for alltwith probability at least 1−
δ
## 2
.  Hence, with
probability 1−δ,
## R(T) =
## ∑
## T
t=1
regret(t) =
## ∑
## T
t=1
regret
## ′
## (t) =
## O
## (
d
## 2
## 
## √
## T
## (1+)
ln(
## 1
δ
) ln(Td)
## )
## .
The  proof  for  the  alternate  definition  of  regret  men-
tioned in Remark 1 is provided in Appendix A.5.
## 4. Conclusions
Detailed concluding remarks appear in supplementary
materials Sec. D.
## References
Abbasi-Yadkori,  Yasin,  P ́al,  D ́avid,  and  Szepesv ́ari,
Csaba.  Improved Algorithms for Linear Stochastic
Bandits.  InNIPS, pp. 2312–2320, 2011.
Abramowitz,  Milton and Stegun, Irene A.Handbook
of  Mathematical  Functions  with  Formulas,  Graphs,
and Mathematical Tables.  Dover, New York, 1964.
Agrawal,   Shipra  and  Goyal,   Navin.Analysis  of
Thompson  Sampling  for  the  Multi-armed  Bandit
Problem.  InCOLT, 2012.
Agrawal, Shipra and Goyal, Navin.  Further Optimal
Regret Bounds for Thompson Sampling.AISTATS,
## 2013.
Auer,Peter.Using    Confidence    Bounds    for
Exploitation-Exploration Trade-offs.Journal of Ma-
chine Learning Research, 3:397–422, 2002.
Auer, Peter, Cesa-Bianchi, Nicol`o, Freund, Yoav, and
## Schapire, Robert E. The Nonstochastic Multiarmed
Bandit  Problem.SIAM  J.  Comput.,  32(1):48–77,
## 2002.
Bubeck, S ́ebastien, Cesa-Bianchi, Nicol`o, and Kakade,
Sham M. Towards minimax policies for online linear
optimization  with  bandit  feedback.Proceedings  of
the  25th  Conference  on  Learning  Theory  (COLT),
pp. 1–14, 2012.

Thompson Sampling for Contextual Bandits with Linear Payoffs
Chapelle, Olivier and Li, Lihong.  An Empirical Eval-
uation of Thompson Sampling.  InNIPS, pp. 2249–
## 2257, 2011.
Chapelle, Olivier and Li, Lihong.  Open Problem:  Re-
gret  Bounds  for  Thompson  Sampling.   InCOLT,
## 2012.
Chu,  Wei,  Li,  Lihong,  Reyzin,  Lev,  and  Schapire,
Robert  E.   Contextual  Bandits  with  Linear  Payoff
Functions.Journal of Machine Learning Research -
## Proceedings Track, 15:208–214, 2011.
Dani,   Varsha,   Hayes,   Thomas   P.,   and   Kakade,
Sham  M.    Stochastic  Linear  Optimization  under
Bandit Feedback.  InCOLT, pp. 355–366, 2008.
Filippi, Sarah, Capp ́e, Olivier, Garivier, Aur ́elien, and
## Szepesv ́ari, Csaba.  Parametric Bandits:  The Gen-
eralized Linear Case.  InNIPS, pp. 586–594, 2010.
Graepel,Thore,Candela,Joaquin    Qui ̃nonero,
Borchert,  Thomas,  and Herbrich,  Ralf.  Web-Scale
Bayesian  Click-Through  rate  Prediction  for  Spon-
sored Search Advertising in Microsoft’s Bing Search
Engine.  InICML, pp. 13–20, 2010.
Granmo, O.-C.  Solving Two-Armed Bernoulli Bandit
Problems  Using  a  Bayesian  Learning  Automaton.
International Journal of Intelligent Computing and
Cybernetics (IJICC), 3(2):207–234, 2010.
Kaelbling,  Leslie  Pack.Associative  Reinforcement
Learning:  Functions in k-DNF.Machine  Learning,
## 15(3):279–298, 1994.
Kaufmann,  Emilie,  Korda,  Nathaniel,  and  Munos,
## R ́emi.    Thompson  Sampling:   An  Optimal  Finite
Time Analysis.ALT, 2012.
Lai,  T.  L.  and  Robbins,  H.Asymptotically  effi-
cient adaptive allocation rules.Advances in Applied
## Mathematics, 6:4–22, 1985.
Langford, John and Zhang, Tong.  The Epoch-Greedy
Algorithm for Multi-armed Bandits with Side Infor-
mation.  InNIPS, 2007.
May,   Benedict  C.  and  Leslie,   David  S.Simula-
tion  studies  in  optimistic  Bayesian  sampling  in
contextual-bandit   problems.Technical   Report
11:02, Statistics Group, Department of Mathemat-
ics, University of Bristol, 2011.
## May,  Benedict  C.,  Korda,  Nathan,  Lee,  Anthony,
and Leslie, David S.  Optimistic Bayesian sampling
in  contextual-bandit  problems.    Technical  Report
11:01, Statistics Group, Department of Mathemat-
ics, University of Bristol, 2011.
Ortega,  Pedro  A.  and  Braun,  Daniel  A.Linearly
Parametrized  Bandits.Journal  of  Artificial  Intel-
ligence Research, 38:475–511, 2010.
Russo,   Daniel   and   Roy,   Benjamin   Van.Learn-
ing  to  optimize  via  posterior  sampling.CoRR,
abs/1301.2609, 2013.
Sarkar, Jyotirmoy. One-armed badit problem with co-
variates.The Annals of Statistics, 19(4):1978–2002,
## 1991.
Scott, S.  A modern Bayesian look at the multi-armed
bandit.Applied  Stochastic  Models  in  Business  and
## Industry, 26:639–658, 2010.
## Strehl,  Alexander  L.,  Mesterharm,  Chris,  Littman,
Michael L.,  and Hirsh,  Haym.  Experience-efficient
learning in associative bandit problems.  InICML,
pp. 889–896, 2006.
Strens, Malcolm J. A.  A Bayesian Framework for Re-
inforcement Learning. InICML, pp. 943–950, 2000.
Thompson,  William  R.   On  the  likelihood  that  one
unknown probability exceeds another in view of the
evidence of two samples.Biometrika, 25(3-4):285–
## 294, 1933.
Woodroofe,  Michael.    A  one-armed  bandit  problem
with a concomitant variable.Journal of the Ameri-
can Statistics Association, 74(368):799–806, 1979.
Wyatt, Jeremy.Exploration and Inference in Learning
from Reinforcement. PhD thesis, Department of Ar-
tificial Intelligence, University of Edinburgh, 1997.