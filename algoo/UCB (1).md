

## Machine Learning, 47, 235–256, 2002
c
2002 Kluwer Academic Publishers. Manufactured in The Netherlands.
Finite-time Analysis of the Multiarmed Bandit
## Problem*
PETER AUERpauer@igi.tu-graz.ac.at
University of Technology Graz, A-8010 Graz, Austria
## NICOL
## `
O CESA-BIANCHIcesa-bianchi@dti.unimi.it
DTI, University of Milan, via Bramante 65, I-26013 Crema, Italy
PAUL FISCHERfischer@ls2.informatik.uni-dortmund.de
Lehrstuhl Informatik II, Universit
## ̈
at Dortmund, D-44221 Dortmund, Germany
Editor:Jyrki Kivinen
Abstract.Reinforcement learning policies face the exploration versus exploitation dilemma, i.e. the search for
a balance between exploring the environment to find profitable actions while taking the empirically best action as
often as possible. A popular measure of a policy’s success in addressing this dilemma is the regret, that is the loss
due to the fact that the globally optimal policy is not followed all the times. One of the simplest examples of the
exploration/exploitation dilemma is the multi-armed bandit problem. Lai and Robbins were the first ones to show
that the regret for this problem has to grow at least logarithmically in the number of plays. Since then, policies
which asymptotically achieve this regret have been devised by Lai and Robbins and many others. In this work we
show that the optimal logarithmic regret is also achievable uniformly over time, with simple and efficient policies,
and for all reward distributions with bounded support.
Keywords:bandit problems, adaptive allocation rules, finite horizon regret
## 1.    Introduction
The exploration versus exploitation dilemma can be described as the search for a balance
between exploring the environment to find profitable actions while taking the empirically
best  action  as  often  as  possible.  The  simplest  instance  of  this  dilemma  is  perhaps  the
multi-armed bandit, a problem extensively studied in statistics (Berry & Fristedt, 1985) that
has also turned out to be fundamental in different areas of artificial intelligence, such as
reinforcement learning (Sutton & Barto, 1998) and evolutionary programming (Holland,
## 1992).
In its most basic formulation, aK-armed bandit problem is defined by random variables
## X
i,n
for 1≤i≤Kandn≥1, where eachiis the index of a gambling machine (i.e., the
“arm” of a bandit). Successive plays of machineiyield rewardsX
i,1
## ,X
i,2
,...which are
## ∗
A preliminary version appeared inProc. of 15th International Conference on Machine Learning, pages 100–108.
## Morgan Kaufmann, 1998

## 236
## P. AUER, N. CESA-BIANCHI AND P. FISCHER
independent and identically distributed according to an unknown law with unknown ex-
pectationμ
i
. Independence also holds for rewards across machines; i.e.,X
i,s
andX
j,t
are
independent (and usually not identically distributed) for each 1≤i<j≤Kand each
s,t≥1.
Apolicy,orallocation strategy,Ais an algorithm that chooses the next machine to play
based on the sequence of past plays and obtained rewards. LetT
i
(n)be the number of times
machineihas been played byAduring thefirstnplays. Then theregretofAafternplays
is defined by
μ
## ∗
n−μ
j
## K
## 
j=1
## IE[T
j
(n)]    whereμ
## ∗
def
## =max
1≤i≤K
μ
i
andIE[·] denotes expectation. Thus the regret is the expected loss due to the fact that the
policy does not always play the best machine.
In their classical paper, Lai and Robbins (1985) found, for specific families of reward
distributions (indexed by a single real parameter), policies satisfying
## IE[T
j
## (n)]≤
## 
## 1
## D(p
j
p
## ∗
## )
## +o(1)
## 
lnn(1)
whereo(1)→0asn→∞and
## D(p
j
p
## ∗
## )
def
## =
## 
p
j
ln
p
j
p
## ∗
is the Kullback-Leibler divergence between the reward densityp
j
of any suboptimal ma-
chinejand the reward densityp
## ∗
of the machine with highest reward expectationμ
## ∗
## .
Hence, under these policies the optimal machine is played exponentially more often than
any other machine, at least asymptotically. Lai and Robbins also proved that this regret is
the best possible. Namely, for any allocation strategy and for any suboptimal machinej,
## IE[T
j
(n)]≥(lnn)/D(p
j
p
## ∗
)asymptotically, provided that the reward distributions satisfy
some mild assumptions.
These policies work by associating a quantity calledupper confidence indexto each ma-
chine. The computation of this index is generally hard. In fact, it relies on the entire sequence
of rewards obtained so far from a given machine. Once the index for each machine is com-
puted, the policy uses it as an estimate for the corresponding reward expectation, picking
for the next play the machine with the current highest index. More recently, Agrawal (1995)
introduced a family of policies where the index can be expressed as simple function of
the total reward obtained so far from the machine. These policies are thus much easier to
compute than Lai and Robbins’, yet their regret retains the optimal logarithmic behavior
(though with a larger leading constant in some cases).
## 1
In this paper we strengthen previous results by showing policies that achieve logarithmic
regret uniformly over time, rather than only asymptotically. Our policies are also simple to
implement and computationally efficient. In Theorem 1 we show that a simple variant of
Agrawal’s index-based policy hasfinite-time regret logarithmically bounded for arbitrary
sets of reward distributions with bounded support (a regret with better constants is proven

## FINITE-TIME ANALYSIS
## 237
in Theorem 2 for a more complicated version of this policy). A similar result is shown
in Theorem 3 for a variant of the well-known randomizedε-greedy heuristic. Finally, in
Theorem 4 we show another index-based policy with logarithmically boundedfinite-time
regret  for  the  natural  case  when  the  reward  distributions  are  normally  distributed  with
unknown means and variances.
Throughout the paper, and whenever the distributions of rewards for each machine are
understood from the context, we define


i
def
## =μ
## ∗
## −μ
i
where, we recall,μ
i
is the reward expectation for machineiandμ
## ∗
is any maximal element
in the set{μ
## 1
## ,...,μ
## K
## }.
-    Main results
Ourfirst result shows that there exists an allocation strategy,
UCB1, achieving logarithmic
regret uniformly overnand without any preliminary knowledge about the reward distri-
butions (apart from the fact that their support is in [0,1]). The policy
UCB1 (sketched in
figure 1) is derived from the index-based policy of Agrawal (1995). The index of this policy
is the sum of two terms. Thefirst term is simply the current average reward. The second term
is related to the size (according to Chernoff-Hoeffding bounds, see Fact 1) of the one-sided
confidence interval for the average reward within which the true expected reward falls with
overwhelming probability.
Theorem 1.For all K>1,if policy
UCB1is run on K machines having arbitrary reward
distributions P
## 1
## ,...,P
## K
with support in[0,1],then its expected regret after any number
n of plays is at most
## 
## 8
## 
i:μ
i
## <μ
## ∗
## 
lnn


i
## 
## 
## +
## 
## 1+
π
## 2
## 3
## 
## 
## K
## 
j=1


j

whereμ
## 1
## ,...,μ
## K
are the expected values of P
## 1
## ,...,P
## K
## .
Figure 1.    Sketch of the deterministic policyUCB1 (see Theorem 1).

## 238
## P. AUER, N. CESA-BIANCHI AND P. FISCHER
Figure 2.    Sketch of the deterministic policy
UCB2 (see Theorem 2).
To prove Theorem 1 we show that, for any suboptimal machinej,
## IE[T
j
## (n)]≤
## 8


## 2
j
lnn(2)
plus a small constant. The leading constant 8/

## 2
i
is worse than the corresponding constant
1/D(p
j
p
## ∗
)in Lai and Robbins’result (1). In fact, one can show thatD(p
j
p
## ∗
## )≥2

## 2
j
where the constant 2 is the best possible.
Using a slightly more complicated policy, which we call
UCB2 (seefigure 2), we can bring
the main constant of (2) arbitrarily close to 1/(2

## 2
j
). The policyUCB2 works as follows.
The plays are divided in epochs. In each new epoch a machineiis picked and then
playedτ(r
i
## +1)−τ(r
i
)times, whereτis an exponential function andr
i
is the number of
epochs played by that machine so far. The machine picked in each new epoch is the one
maximizing ̄x
i
## +a
n,r
i
, wherenis the current number of plays, ̄x
i
is the current average
reward for machinei, and
a
n,r
## =


## (1+α)ln(en/τ(r))
## 2τ(r)
## (3)
where
τ(r)=(1+α)
r
## .
In the next result we state a bound on the regret of
UCB2. The constantc
α
, here left unspec-
ified, is defined in (18) in the appendix, where the theorem is also proven.
Theorem 2.For all K>1,if policy
UCB2is run with input0<α<1on K  machines
having arbitrary reward distributions P
## 1
## ,...,P
## K
with support in[0,1],then its expected
regret after any number
n≥max
i:μ
i
## <μ
## ∗
## 1
## 2

## 2
i

## FINITE-TIME ANALYSIS239
of plays is at most
## 
i:μ
i
## <μ
## ∗
## 
## (1+α)(1+4α)ln

## 2e

## 2
i
n

## 2

i
## +
c
α


i
## 
## (4)
whereμ
## 1
## ,...,μ
## K
are the expected values of P
## 1
## ,...,P
## K
## .
Remark.    By choosingαsmall, the constant of the leading term in the sum (4) gets arbi-
trarily close to 1/(2

## 2
i
); however,c
α
→∞asα→0. The two terms in the sum can be
traded-off by lettingα=α
n
be slowly decreasing with the numbernof plays.
A  simple  and  well-known  policy  for  the  bandit  problem  is  the  so-calledε-greedy  rule
(see Sutton, & Barto, 1998). This policy prescribes to play with probability 1−εthe machine
with the highest average reward, and with probabilityεa randomly chosen machine. Clearly,
the constant exploration probabilityεcauses a linear (rather than logarithmic) growth in
the regret. The obviousfixistoletεgo to zero with a certain rate, so that the exploration
probability decreases as our estimates for the reward expectations become more accurate.
It turns out that a rate of 1/n, wherenis, as usual, the index of the current play, allows
to prove a logarithmic bound on the regret. The resulting policy,ε
n
-GREEDY, is shown in
figure 3.
Theorem 3.For all K>1and for all reward distributions  P
## 1
## ,...,P
## K
with support in
[0,1],if policyε
n
## -GREEDY
is run with input parameter
## 0<d≤min
i:μ
i
## <μ
## ∗


i
## ,
Figure 3.    Sketch of the randomized policyε
n
-GREEDY(see Theorem 3).

## 240P. AUER, N. CESA-BIANCHI AND P. FISCHER
then the probability that after any number n≥cK/d of playsε
n
-GREEDYchooses a subop-
timal machine  j is at most
c
d
## 2
n
## +2
## 
c
d
## 2
ln
## (n−1)d
## 2
e
## 1/2
cK
## 
cK
## (n−1)d
## 2
e
## 1/2
## 
c/(5d
## 2
## )
## +
## 4e
d
## 2
## 
cK
## (n−1)d
## 2
e
## 1/2
## 
c/2
## .
Remark.Forclarge enough (e.g.c>5) the above bound is of orderc/(d
## 2
n)+o(1/n)for
n→∞, as the second and third terms in the bound areO(1/n
## 1+ε
)for someε>0 (recall
that 0<d<1). Note also that this is a result stronger than those of Theorems 1 and 2, as
it establishes a bound on the instantaneous regret. However, unlike Theorems 1 and 2, here
we need to know a lower bounddon the difference between the reward expectations of the
best and the second best machine.
Our last result concerns a special case, i.e. the bandit problem with normally distributed
rewards. Surprisingly, we could notfind in the literature regret bounds (not even asymp-
totical) for the case when both the mean and the variance of the reward distributions are
unknown. Here, we show that an index-based policy called
UCB1-NORMAL, seefigure 4,
achieves logarithmic regret uniformly overnwithout knowing means and variances of the
reward distributions. However, our proof is based on certain bounds on the tails of theχ
## 2
and the Student distribution that we could only verify numerically. These bounds are stated
as Conjecture 1 and Conjecture 2 in the Appendix.
The choice of the index in
UCB1-NORMALis based, as forUCB1, on the size of the one-
sided confidence interval for the average reward within which the true expected reward falls
with overwhelming probability. In the case of
UCB1, the reward distribution was unknown,
and we used Chernoff-Hoeffding bounds to compute the index. In this case we know that
Figure 4.    Sketch of the deterministic policyUCB1-NORMAL(see Theorem 4).

## FINITE-TIME ANALYSIS
## 241
the distribution is normal, and for computing the index we use the sample variance as an
estimate of the unknown variance.
Theorem 4.For all K>1,if policy
UCB1-NORMALis run on K machines having normal
reward distributions P
## 1
## ,...,P
## K
,then its expected regret after any number n of plays is at
most
## 256(logn)
## 
## 
i:μ
i
## <μ
## ∗
σ
## 2
i


i

## +
## 
## 1+
π
## 2
## 2
+8 logn
## 
## K
## 
j=1


j

whereμ
## 1
## ,...,μ
## K
andσ
## 2
## 1
## ,...,σ
## 2
## K
are  the  means  and  variances  of  the  distributions
## P
## 1
## ,...,P
## K
## .
As afinal remark for this section, note that Theorems 1–3 also hold for rewards that are not
independent across machines, i.e.X
i,s
andX
j,t
might be dependent for anys,t, andi=j.
Furthermore, we also do not need that the rewards of a single arm are i.i.d., but only the
weaker assumption thatIE[X
i,t
## |X
i,1
## ,...,X
i,t−1
## ]=μ
i
for all 1≤t≤n.
## 3.    Proofs
Recall that, for each 1≤i≤K,IE[X
i,n
## ]=μ
i
for alln≥1 andμ
## ∗
## =max
1≤i≤K
μ
i
## . Also,
for anyfixed policyA,T
i
(n)is the number of times machineihas been played byAin the
firstnplays. Of course, we always have

## K
i=1
## T
i
(n)=n. We also define the r.v.’sI
## 1
## ,I
## 2
## ,...,
whereI
t
denotes the machine played at timet.
For each 1≤i≤Kandn≥1define
## ̄
## X
i,n
## =
## 1
n
n
## 
t=1
## X
i,t
## .
## Givenμ
## 1
## ,...,μ
## K
, we calloptimalthe machine with the least indexisuch thatμ
i
## =μ
## ∗
## .
In what follows, we will always put a superscript“
## ∗
”to any quantity which refers to the
optimal machine. For example we writeT
## ∗
## (n)and
## ̄
## X
## ∗
n
instead ofT
i
## (n)and
## ̄
## X
i,n
, whereiis
the index of the optimal machine.
Some further notation: For any predicatewe define{(x)}to be the indicator fuction
of the event(x); i.e.,{(x)}=1if(x)is true and{(x)}=0 otherwise. Finally,
Va r
## [
## X
## ]
denotes the variance of the random variableX.
Note that the regret afternplays can be written as
## 
j:μ
j
## <μ
## ∗


j
## IE[T
j
## (n)](5)
So we can bound the regret by simply bounding eachIE[T
j
## (n)].
We will make use of the following standard exponential inequalities for bounded random
variables (see, e.g., the appendix of Pollard, 1984).

## 242
## P. AUER, N. CESA-BIANCHI AND P. FISCHER
Fact 1(Chernoff-Hoeffding bound).Let X
## 1
## ,...,X
n
be random variables with common
range[0,1]and such that IE[X
t
## |X
## 1
## ,...,X
t−1
## ]=μ. Let S
n
## =X
## 1
## +···+X
n
. Then for
all a≥0
## IP{S
n
## ≥nμ+a}≤e
## −2a
## 2
## /n
andIP{S
n
## ≤nμ−a}≤e
## −2a
## 2
## /n
Fact 2(Bernstein inequality).Let X
## 1
## ,...,X
n
be random variables with range[0,1]and
n
## 
t=1
Va r
## [
## X
t
## |X
t−1
## ,...,X
## 1
## ]
## =σ
## 2
## .
## Let S
n
## =X
## 1
## +···+X
n
. Then for all a≥0
## IP{S
n
## ≥IE[S
n
## ]+a}≤exp
## 
## −
a
## 2
## /2
σ
## 2
## +a/2
## 
## .
Proof of Theorem 1:Letc
t,s
## =
## √
(2lnt)/s. For any machinei, we upper boundT
i
## (n)
on any sequence of plays. More precisely, for eacht≥1 we bound the indicator function
ofI
t
=ias follows. Letbe an arbitrary positive integer.
## T
i
## (n)=1+
n
## 
t=K+1
## {I
t
## =i}
## ≤+
n
## 
t=K+1
## {I
t
=i,T
i
## (t−1)≥}
## ≤+
n
## 
t=K+1
## 
## ̄
## X
## ∗
## T
## ∗
## (t−1)
## +c
t−1,T
## ∗
## (t−1)
## ≤
## ̄
## X
i,T
i
## (t−1)
## +c
t−1,T
i
## (t−1)
## ,T
i
## (t−1)≥
## 
## ≤+
n
## 
t=K+1
## 
min
## 0<s<t
## ̄
## X
## ∗
s
## +c
t−1,s
## ≤max
## ≤s
i
## <t
## ̄
## X
i,s
i
## +c
t−1,s
i
## 
## ≤+
## ∞
## 
t=1
t−1
## 
s=1
t−1
## 
s
i
## =
## 
## ̄
## X
## ∗
s
## +c
t,s
## ≤
## ̄
## X
i,s
i
## +c
t,s
i
## 
## .(6)
Now observe that
## ̄
## X
## ∗
s
## +c
t,s
## ≤
## ̄
## X
i,s
i
## +c
t,s
i
implies that at least one of the following must
hold
## ̄
## X
## ∗
s
## ≤μ
## ∗
## −c
t,s
## (7)
## ̄
## X
i,s
i
## ≥μ
i
## +c
t,s
i
## (8)
μ
## ∗
## <μ
i
## +2c
t,s
i
## .(9)
We bound the probability of events (7) and (8) using Fact 1 (Chernoff-Hoeffding bound)
## IP{
## ̄
## X
## ∗
s
## ≤μ
## ∗
## −c
t,s
## }≤e
## −4lnt
## =t
## −4

## FINITE-TIME ANALYSIS
## 243
## IP
## 
## ̄
## X
i,s
i
## ≥μ
i
## +c
t,s
i
## 
## ≤e
## −4lnt
## =t
## −4
## .
## For=(8lnn)/

## 2
i
, (9) is false. In fact
μ
## ∗
## −μ
i
## −2c
t,s
i
## =μ
## ∗
## −μ
i
## −2
## 
## 2(lnt)/s
i
## ≥μ
## ∗
## −μ
i
## −

i
## =0
fors
i
## ≥(8lnn)/

## 2
i
.Soweget
## IE[T
i
## (n)]≤
## 
## 8lnn


## 2
i
## 
## +
## ∞
## 
t=1
t−1
## 
s=1
t−1
## 
s
i
## =
## 
## (8lnn)/

## 2
i
## 
## ×

## IP{
## ̄
## X
## ∗
s
## ≤μ
## ∗
## −c
t,s
## }+IP
## 
## ̄
## X
i,s
i
## ≥μ
i
## +c
t,s
i
## 
## ≤
## 
## 8lnn


## 2
i
## 
## +
## ∞
## 
t=1
t
## 
s=1
t
## 
s
i
## =1
## 2t
## −4
## ≤
## 8lnn


## 2
i
## +1+
π
## 2
## 3
which concludes the proof.✷
Proof  of Theorem 3:Recall that, forn≥cK/d
## 2
## ,ε
n
=cK/(d
## 2
n). Let
x
## 0
## =
## 1
## 2K
n
## 
t=1
ε
t
## .
The probability that machinejis chosen at timenis
## IP{I
n
## =j}≤
ε
n
## K
## +
## 
## 1−
ε
n
## K
## 
## IP
## 
## ̄
## X
j,T
j
## (n−1)
## ≥
## ̄
## X
## ∗
## T
## ∗
## (n−1)
## 
and
## IP
## 
## ̄
## X
j,T
j
## (n)
## ≥
## ̄
## X
## ∗
## T
## ∗
## (n)
## 
## ≤IP
## 
## ̄
## X
j,T
j
## (n)
## ≥μ
j
## +


j
## 2
## 
## +IP
## 
## ̄
## X
## ∗
## T
## ∗
## (n)
## ≤μ
## ∗
## −


j
## 2
## 
## .(10)
Now the analysis for both terms on the right-hand side is the same. LetT
## R
j
(n)be the number
of plays in which machinejwas chosenat randomin thefirstnplays. Then we have
## IP
## 
## ̄
## X
j,T
j
## (n)
## ≥μ
j
## +


j
## 2
## 
## =
n
## 
t=1
## IP
## 
## T
j
## (n)=t∧
## ̄
## X
j,t
## ≥μ
j
## +


j
## 2
## 
## (11)

## 244
## P. AUER, N. CESA-BIANCHI AND P. FISCHER
## =
n
## 
t=1
## IP
## 
## T
j
## (n)=t|
## ̄
## X
j,t
## ≥μ
j
## +


j
## 2
## 
## ·IP
## 
## ̄
## X
j,t
## ≥μ
j
## +


j
## 2
## 
## ≤
n
## 
t=1
## IP
## 
## T
j
## (n)=t|
## ̄
## X
j,t
## ≥μ
j
## +


j
## 2
## 
## ·e
## −

## 2
j
t/2
by Fact 1 (Chernoff-Hoeffding bound)
## ≤
## x
## 0
## 
## 
t=1
## IP
## 
## T
j
## (n)=t|
## ̄
## X
j,t
## ≥μ
j
## +


j
## 2
## 
## +
## 2


## 2
j
e
## −

## 2
j
## x
## 0
## /2
since

## ∞
t=x+1
e
## −κt
## ≤
## 1
κ
e
## −κx
## ≤
## x
## 0
## 
## 
t=1
## IP
## 
## T
## R
j
## (n)≤t|
## ̄
## X
j,t
## ≥μ
j
## +


j
## 2
## 
## +
## 2


## 2
j
e
## −

## 2
j
## x
## 0
## /2
## ≤x
## 0
## ·IP
## 
## T
## R
j
## (n)≤x
## 0
## 
## +
## 2


## 2
j
e
## −

## 2
j
## x
## 0
## /2
## (12)
where in the last line we dropped the conditioning because each machine is played at random
independently of the previous choices of the policy. Since
## IE
## 
## T
## R
j
## (n)
## 
## =
## 1
## K
n
## 
t=1
ε
t
and
Va r
## 
## T
## R
j
## (n)
## 
## =
n
## 
t=1
ε
t
## K
## 
## 1−
ε
t
## K
## 
## ≤
## 1
## K
n
## 
t=1
ε
t
## ,
by Bernstein’s inequality (2) we get
## IP
## 
## T
## R
j
## (n)≤x
## 0
## 
## ≤e
## −x
## 0
## /5
## .(13)
Finally it remains to lower boundx
## 0
.Forn≥n
## 
=cK/d
## 2
## ,ε
n
=cK/(d
## 2
n)and we have
x
## 0
## =
## 1
## 2K
n
## 
t=1
ε
t
## =
## 1
## 2K
n
## 
## 
t=1
ε
t
## +
## 1
## 2K
n
## 
t=n
## 
## +1
ε
t
## ≥
n
## 
## 2K
## +
c
d
## 2
ln
n
n
## 
## ≥
c
d
## 2
ln
nd
## 2
e
## 1/2
cK
## .

## FINITE-TIME ANALYSIS
## 245
Thus, using (10)–(13) and the above lower bound onx
## 0
we obtain
## IP{I
n
## =j}≤
ε
n
## K
## +2x
## 0
e
## −x
## 0
## /5
## +
## 4


## 2
j
e
## −

## 2
j
## x
## 0
## /2
## ≤
c
d
## 2
n
## +2
## 
c
d
## 2
ln
## (n−1)d
## 2
e
## 1/2
cK
## 
cK
## (n−1)d
## 2
e
## 1/2
## 
c/(5d
## 2
## )
## +
## 4e
d
## 2
## 
cK
## (n−1)d
## 2
e
## 1/2
## 
c/2
## .
This concludes the proof.✷
## 4.    Experiments
For practical purposes, the bound of Theorem 1 can be tuned morefinely. We use
## V
j
## (s)
def
## =
## 
## 1
s
s
## 
τ=1
## X
## 2
j,τ

## −
## ̄
## X
## 2
j,s
## +
## 
## 2lnt
s
as un upper confidence bound for the variance of machinej. As before, this means that
machinej, which has been playedstimes during thefirsttplays, has a variance that is
at most the sample variance plus
## √
(2lnt)/s. We then replace the upper confidence bound
## 
## 2ln(n)/n
j
of policy
## UCB
1 with


lnn
n
j
min{1/4,V
j
## (n
j
## )}
(the factor 1/4 is an upper bound on the variance of a Bernoulli random variable). This
variant, which we call
UCB1-TUNED, performs substantially better thanUCB1 in essentially
all of our experiments. However, we are not able to prove a regret bound.
We compared the empirical behaviour policies
UCB1-TUNED,UCB2, andε
n
-GREEDYon
Bernoulli reward distributions with different parameters shown in the table below.
## 12345678910
## 10.90.6
## 20.90.8
## 30.550.45
## 110.90.60.60.60.60.60.60.60.60.6
## 120.90.80.80.80.70.70.70.60.60.6
## 130.90.80.80.80.80.80.80.80.80.8
## 140.550.450.450.450.450.450.450.450.450.45

## 246P. AUER, N. CESA-BIANCHI AND P. FISCHER
Rows 1–3define reward distributions for a 2-armed bandit problem, whereas rows 11–
14  define  reward  distributions  for  a  10-armed  bandit  problem.  The  entries  in  each  row
denote the reward expectations (i.e. the probabilities of getting a reward 1, as we work with
Bernoulli distributions) for the machines indexed by the columns. Note that distributions 1
and 11 are“easy”(the reward of the optimal machine has low variance and the differences
μ
## ∗
## −μ
i
are all large), whereas distributions 3 and 14 are“hard”(the reward of the optimal
machine has high variance and some of the differencesμ
## ∗
## −μ
i
are small).
We made experiments to test the different policies (or the same policy with different
input parameters) on the seven distributions listed above. In each experiment we tracked
two performance measures: (1) the percentage of plays of the optimal machine; (2) the
actual regret, that is the difference between the reward of the optimal machine and the
reward of the machine played. The plot for each experiment shows, on a semi-logarithmic
scale, the behaviour of these quantities during 100,000 plays averaged over 100 different
runs. We ran afirst round of experiments on distribution 2 tofind out good values for the
parameters of the policies. If a parameter is chosen too small, then the regret grows linearly
(exponentially in the semi-logarithmic plot); if a parameter is chosen too large then the
regret grows logarithmically, but with a large leading constant (corresponding to a steep
line in the semi-logarithmic plot).
## Policy
UCB2 is relatively insensitive to the choice of its parameterα, as long as it is
kept relatively small (seefigure 5). Afixed value 0.001 has been used for all the remaining
experiments. On other hand, the choice ofcin policyε
n
-GREEDYis difficult as there is no
value that works reasonably well for all the distributions that we considered. Therefore, we
have roughly searched for the best value for each distribution. In the plots, we will also
show the performance ofε
n
-GREEDYfor values ofcaround this empirically best value. This
shows that the performance degrades rapidly if this parameter is not appropriately tuned.
Finally, in each experiment the parameterdofε
n
-GREEDYwas set to

## =μ
## ∗
## −max
i:μ
i
## <μ
## ∗
μ
i
## .
Figure 5.    Search for the best value of parameterαof policyUCB2.

## FINITE-TIME ANALYSIS
## 247
4.1.    Comparison between policies
We can summarize the comparison of all the policies on the seven distributions as follows
## (see Figs. 6–12).
–An optimally tunedε
n
-GREEDYperforms almost always best. Significant exceptions are
distributions 12 and 14: this is becauseε
n
## -GREEDY
explores uniformly over all machines,
thus the policy is hurt if there are several nonoptimal machines, especially when their
reward expectations differ a lot. Furthermore, ifε
n
-GREEDYis not well tuned its perfor-
mance degrades rapidly (except for distribution 13, on whichε
n
-GREEDYperforms well
a wide range of values of its parameter).
–In most cases,
UCB1-TUNEDperforms comparably to a well-tunedε
n
-GREEDY. Further-
more,
## UCB1-
TUNEDis not very sensitive to the variance of the machines, that is why it
performs similarly on distributions 2 and 3, and on distributions 13 and 14.
–Policy
UCB2 performs similarly toUCB1-TUNED, but always slightly worse.
Figure 6.    Comparison on distribution 1 (2 machines with parameters 0.9,0.6).
Figure 7.    Comparison on distribution 2 (2 machines with parameters 0.9,0.8).

## 248P. AUER, N. CESA-BIANCHI AND P. FISCHER
Figure 8.    Comparison on distribution 3 (2 machines with parameters 0.55,0.45).
Figure 9.    Comparison on distribution 11 (10 machines with parameters 0.9,0.6,...,0.6).
Figure 10.    Comparison on distribution 12 (10 machines with parameters 0.9,0.8,0.8,0.8,0.7,0.7,0.7,0.6,
## 0.6,0.6).

## FINITE-TIME ANALYSIS
## 249
Figure 11.    Comparison on distribution 13 (10 machines with parameters 0.9,0.8,...,0.8).
Figure 12.    Comparison on distribution 14 (10 machines with parameters 0.55,0.45,...,0.45).
## 5.    Conclusions
We have shown simple and efficient policies for the bandit problem that, on any set of reward
distributions with known bounded support, exhibit uniform logarithmic regret. Our policies
are deterministic and based on upper confidence bounds, with the exception ofε
n
## -GREEDY,
a randomized allocation rule that is a dynamic variant of theε-greedy heuristic. Moreover,
our policies are robust with respect to the introduction of moderate dependencies in the
reward processes.
This work can be extended in many ways. A more general version of the bandit problem
is obtained by removing the stationarity assumption on reward expectations (see Berry &
Fristedt,  1985;  Gittins,  1989  for  extensions  of  the  basic  bandit  problem).  For  example,
suppose that a stochastic reward process{X
i,s
:s=1,2,...}is associated to each machine
i=1,...,K. Here, playing machineiat timetyields a rewardX
i,s
and causes the current

## 250
## P. AUER, N. CESA-BIANCHI AND P. FISCHER
statesofito change tos+1, whereas the states of other machines remain frozen. A well-
studied problem in this setup is the maximization of the total expected reward in a sequence
ofnplays. There are methods, like the Gittins allocation indices, that allow tofind the
optimal machine to play at each timenby considering each reward process independently
from the others (even though the globally optimal solution depends on all the processes).
However, computation of the Gittins indices for the average (undiscounted) reward criterion
used here requires preliminary knowledge about the reward processes (see, e.g., Ishikida &
Varaiya, 1994). To overcome this requirement, one can learn the Gittins indices, as proposed
in Duff (1995) for the case offinite-state Markovian reward processes. However, there are no
finite-time regret bounds shown for this solution. At the moment, we do not know whether
our techniques could be extended to these more general bandit problems.
Appendix A:    Proof of Theorem 2
Note that
τ(r)≤(1+α)
r
## +1≤τ(r−1)(1+α)+1(14)
forr≥1. Assume thatn≥1/(2

## 2
j
)for alljand let ̃r
j
be the largest integer such that
τ( ̃r
j
## −1)≤
## (1+4α)ln

## 2en

## 2
j

## 2

## 2
j
## .
Note that ̃r
j
≥1. We have
## T
j
## (n)≤1+
## 
r≥1
(τ(r)−τ(r−1)){machinejfinishes itsr-th epoch}
## ≤τ( ̃r
j
## )+
## 
r> ̃r
j
(τ(r)−τ(r−1)){machinejfinishes itsr-th epoch}
Now consider the following chain of implications
machinejfinishes itsr-th epoch
⇒∃i≥0,∃t≥τ(r−1)+τ(i)such that

## ̄
## X
j,τ(r−1)
## +a
t,r−1

## ≥

## ̄
## X
## ∗
τ(i)
## +a
t,i

⇒∃t≥τ(r−1)such that

## ̄
## X
j,τ(r−1)
## +a
t,r−1

## ≥μ
## ∗
## −α

j
## /2
or∃i≥0,∃t
## 
≥τ(r−1)+τ(i)such that

## ̄
## X
## ∗
τ(i)
## +a
t
## 
## ,i

## ≤μ
## ∗
## −α

j
## /2
## ⇒
## ̄
## X
j,τ(r−1)
## +a
n,r−1
## ≥μ
## ∗
## −α

j
## /2
or∃i≥0 such that
## ̄
## X
## ∗
τ(i)
## +a
τ(r−1)+τ(i),i
## ≤μ
## ∗
## −α

j
## /2
where the last implication hold becausea
t,r
is increasing int. Hence
## IE[T
j
## (n)]≤τ( ̃r
j
## )+
## 
r> ̃r
j
(τ(r)−τ(r−1))IP
## 
## ̄
## X
j,τ(r−1)
## +a
n,r−1
## ≥μ
## ∗
## −α

j
## /2
## 

## FINITE-TIME ANALYSIS
## 251
## +
## 
r> ̃r
j
## 
i≥0
## (τ(r)−τ(r−1))
## ·IP
## 
## ̄
## X
## ∗
τ(i)
## +a
τ(r−1)+τ(i),i
## ≤μ
## ∗
## −α

j
## /2
## 
## .(15)
The assumptionn≥1/(2

## 2
j
)implies ln(2en

## 2
j
)≥1. Therefore, forr> ̃r
j
## ,wehave
τ(r−1)>
## (1+4α)ln

## 2en

## 2
j

## 2

## 2
j
## (16)
and
a
n,r−1
## =


## (1+α)ln(en/τ(r−1))
## 2τ(r−1)
## ≤

j


## (1+α)ln(en/τ(r−1))

## 1+4α)ln(2en

## 2
j

using (16) above
## ≤

j
## 
## 
## 
## 

## 1+α)ln(2en

## 2
j


## 1+4α)ln(2en

## 2
j

usingτ(r−1)>1/2

## 2
j
derived from (16)
## ≤

j
## 
## 1+α
## 1+4α
## .(17)
We start by bounding thefirst sum in (15). Using (17) and Fact 1 (Chernoff-Hoeffding
bound) we get
## IP
## 
## ̄
## X
j,τ(r−1)
## +a
n,r−1
## ≥μ
## ∗
## −α

j
## /2
## 
## =IP{
## ̄
## X
j,τ(r−1)
## +a
n,r−1
## ≥μ
j
## +

j
## −α

j
## /2}
## ≤exp
## 
## −2τ(r−1)

## 2
j

## 1−α/2−
## 
## (1+α)/(1+4α)

## 2
## 
## ≤exp
## 
## −2τ(r−1)

## 2
j
## (
## 1−α/2−(1−α)
## )
## 2
## 
## =exp
## 
## −τ(r−1)

## 2
j
α
## 2
## 
## 2
## 
forα<1/10. Now letg(x)=(x−1)/(1+α). By (14) we getg(x)≤τ(r−1)for
τ(r−1)≤x≤τ(r)andr≥1. Hence
## 
r> ̃r
j
(τ(r)−τ(r−1))IP
## 
## ̄
## X
j,τ(r−1)
## +a
n,r−1
## ≥μ
## ∗
## −α

j
## /2
## 
## ≤
## 
r> ̃r
j
## (τ(r)−τ(r−1))exp
## 
## −τ(r−1)

## 2
j
α
## 2
## 
## ≤
## 
## ∞
## 0
e
## −cg(x)
dx

## 252
## P. AUER, N. CESA-BIANCHI AND P. FISCHER
wherec=(

j
α)
## 2
<1. Further manipulation yields
## 
## ∞
## 0
exp
## 
## −
c
## 1+α
## (x−1)
## 
dx=e
c/(1+α)
## 1+α
c
## ≤
## (1+α)e
## (

j
α)
## 2
## .
We continue by bounding the second sum in (15). Using once more Fact 1, we get
## 
r> ̃r
j
## 
i≥0
(τ(r)−τ(r−1))IP
## 
## ̄
## X
## ∗
τ(i)
## +a
τ(r−1)+τ(i),i
## ≤μ
## ∗
## −α

j
## /2
## 
## ≤
## 
i≥0
## 
r> ̃r
j
## (τ(r)−τ(r−1))
## ·exp
## 
## −τ(i)
## (α

j
## )
## 2
## 2
## −(1+α)ln
## 
e
τ(r−1)+τ(i)
τ(i)
## 
## ≤
## 
i≥0
exp{−τ(i)(α

j
## )
## 2
## /2}
## ·
## 
## 
r> ̃r
j
## (τ(r)−τ(r−1))exp
## 
## −(1+α)ln
## 
## 1+
τ(r−1)
τ(i)
## 
## 
## =
## 
i≥0
exp{−τ(i)(α

j
## )
## 2
## /2}
## ·
## 
## 
r> ̃r
j
## (τ(r)−τ(r−1))
## 
## 1+
τ(r−1)
τ(i)
## 
## −(1+α)
## 
## ≤
## 
i≥0
exp{−τ(i)(α

j
## )
## 2
## /2}
## 
## 
## ∞
## 0
## 
## 1+
x−1
## (1+α)τ(i)
## 
## −(1+α)
dx
## 
## =
## 
i≥0
τ(i)exp{−τ(i)(α

j
## )
## 2
## /2}
## 
## 1+α
α
## 
## 1−
## 1
## (1+α)τ(i)
## 
## −α
## 
## ≤
## 
i≥0
τ(i)exp{−τ(i)(α

j
## )
## 2
## /2}
## 
## 1+α
α
## 
α
## 1+α
## 
## −α
## 
asτ(i)≥1
## =
## 
## 1+α
α
## 
## 1+α
## 
i≥0
τ(i)exp{−τ(i)(α

j
## )
## 2
## /2}.
Now, as(1+α)
x−1
## ≤τ(i)≤(1+α)
x
+1 fori≤x≤i+1, we can bound the series in
the last formula above with an integral
## 
i≥0
τ(i)exp{−τ(i)(α

j
## )
## 2
## /2}
## ≤1+
## 
## ∞
## 1
## ((1+α)
x
## +1)exp{−(1+α)
x−1
## (α

j
## )
## 2
## /2}dx

## FINITE-TIME ANALYSIS
## 253
## ≤1+
## 
## ∞
## 1
z+1
zln(1+α)
exp
## 
## −
z(α

j
## )
## 2
## 2(1+α)
## 
dz
by change of variablez=(1+α)
x
## =1+
## 1
ln(1+α)
## 
e
## −λ
λ
## +
## 
## ∞
λ
e
## −x
x
dx
## 
where we set
λ=
## (α

j
## )
## 2
## 2(1+α)
## .
## As 0<α,

j
<1, we have 0<λ<1/4. To upper bound the bracketed formula above,
consider the function
## F(λ)=e
## −λ
## +λ
## 
## ∞
λ
e
## −x
x
dx
with derivatives
## F
## 
## (λ)=
## 
## ∞
λ
e
## −x
x
dx−2e
## −λ
## F
## 
## (λ)=2λe
## −λ
## −
## 
## ∞
λ
e
## −x
x
dx.
In the interval(0,1/4),F
## 
is seen to have a zero atλ=0.0108....AsF
## 
(λ)<0 in the
same interval, this is the unique maximum ofF, and wefindF(0.0108...)<11/10. So
we have
e
## −λ
λ
## +
## 
## ∞
λ
e
## −x
x
dx<
## 11
## 10λ
## =
## 11(1+α)
## 5(α

j
## )
## 2
Piecing everything together, and using (14) to upper boundτ( ̃r
j
),wefind that
## IE[T
j
## (n)]≤τ( ̃r
j
## )+
## (1+α)e
## (

j
α)
## 2
## +
## 
## 1+α
α
## 
## 1+α
## 
## 1+
## 11(1+α)
## 5(α

j
## )
## 2
ln(1+α)
## 
## ≤
## (1+α)(1+4α)ln

## 2en

## 2
j

## 2

## 2
j
## +
c
α


## 2
j
where
c
α
## =1+
## (1+α)e
α
## 2
## +
## 
## 1+α
α
## 
## 1+α
## 
## 1+
## 11(1+α)
## 5α
## 2
ln(1+α)
## 
## .(18)
This concludes the proof.✷

## 254
## P. AUER, N. CESA-BIANCHI AND P. FISCHER
Appendix B:    Proof of Theorem 4
The proof goes very much along the same lines as the proof of Theorem 1. It is based on
the two following conjectures which we only verified numerically.
Conjecture 1.LetXbe a Student random variable withsdegrees of freedom. Then, for
all 0≤a≤
## √
## 2(s+1),
IP{X≥a}≤e
## −a
## 2
## /4
## .
Conjecture 2.LetXbe aχ
## 2
random variable withsdegrees of freedom. Then
IP{X≥4s}≤e
## −(s+1)/2
## .
We now proceed with the proof of Theorem 4. Let
## Q
i,n
## =
n
## 
t=1
## X
## 2
i,t
## .
Fix a machineiand, for anysandt, set
c
t,s
## =


## 16·
## Q
i,s
## −s
## ̄
## X
## 2
i,s
s−1
## ·
lnt
s
## Letc
## ∗
t,s
be the corresponding quantity for the optimal machine. To upper boundT
i
## (n),we
proceed exactly as in thefirst part of the proof of Theorem 1 obtaining, for any positive
integer,
## T
i
## (n)≤+
## ∞
## 
t=1
t−1
## 
s=1
t−1
## 
s
i
## =
## 
## {
## ̄
## X
## ∗
s
## ≤μ
## ∗
## −c
## ∗
t,s
## }+
## 
## ̄
## X
i,s
i
## ≥μ
i
## +c
t,s
i
## 
## +
## 
μ
## ∗
## <μ
i
## +2c
t,s
i
## 
## .
The random variable(
## ̄
## X
i,s
i
## −μ
i
## )/

## (Q
i,s
i
## −s
i
## ̄
## X
## 2
i,s
i
## )/(s
i
## (s
i
−1))has a Student distribution
withs
i
−1 degrees of freedom (see, e.g., Wilks, 1962, 8.4.3 page 211). Therefore, using
Conjecture 1 withs=s
i
−1 anda=4
## √
lnt,weget
## IP
## 
## ̄
## X
i,s
i
## ≥μ
i
## +c
t,s
i
## 
## =IP
## 
## 
## 
## ̄
## X
i,s
i
## −μ
i


## Q
i,s
i
## −s
i
## ̄
## X
## 2
i,s
i

## /(s
i
## (s
i
## −1))
## ≥4
## √
lnt
## 
## 
## 
## ≤t
## −4
for alls
i
≥8lnt. The probability of
## ̄
## X
## ∗
s
## ≤μ
## ∗
## −c
## ∗
t,s
is bounded analogously. Finally, since
## (Q
i,s
i
## −s
i
## ̄
## X
## 2
i,s
i
## )/σ
## 2
i
isχ
## 2
-distributed withs
i
−1 degrees of freedom (see, e.g., Wilks, 1962,

## FINITE-TIME ANALYSIS
## 255
8.4.1 page 208). Therefore, using Conjecture 2 withs=s
i
−1 anda=4s,weget
## IP
## 
μ
## ∗
## <μ
i
## +2c
t,s
i
## 
## =IP
## 

## Q
i,s
i
## −s
i
## ̄
## X
## 2
i,s
i
## 
σ
## 2
i
## >(s
i
## −1)


## 2
i
σ
## 2
i
s
i
64 lnt
## 
## ≤IP
## 
## Q
i,s
i
## −s
i
## ̄
## X
## 2
i,s
i
## 
σ
## 2
i
## >4(s
i
## −1)
## 
## ≤e
## −s
i
## /2
## ≤t
## −4
for
s
i
## ≥max
## 
## 256
σ
## 2
i


## 2
i
## ,8
## 
lnt.
## Setting
## =
## 
max
## 
## 256
σ
## 2
i


## 2
i
## ,8
## 
lnt
## 
completes the proof of the theorem.✷
## Acknowledgments
The support from ESPRIT Working Group EP 27150, Neural and Computational Learning
II (NeuroCOLT II), is gratefully acknowledged.
## Note
-  Similar extensions of Lai and Robbins’results were also obtained by Yakowitz and Lowe (1991), and by
Burnetas and Katehakis (1996).
## References
Agrawal, R. (1995). Sample mean based index policies withO(logn)regret for the multi-armed bandit problem.
Advances in Applied Probability,27, 1054–1078.
Berry, D., & Fristedt, B. (1985).Bandit problems. London: Chapman and Hall.
Burnetas, A., & Katehakis, M. (1996). Optimal adaptive policies for sequential allocation problems.Advances in
## Applied Mathematics,17:2, 122–142.
Duff, M. (1995). Q-learning for bandit problems. InProceedings of the 12th International Conference on Machine
## Learning(pp. 209–217).
Gittins, J. (1989).Multi-armed bandit allocation indices, Wiley-Interscience series in Systems and Optimization.
New York: John Wiley and Sons.
Holland, J. (1992).Adaptation in natural and artificial systems. Cambridge: MIT Press/Bradford Books.
Ishikida, T., & Varaiya, P. (1994). Multi-armed bandit problem revisited.Journal of Optimization Theory and
## Applications,83:1, 113–154.
Lai, T., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules.Advances in Applied Mathematics,
## 6,4–22.
Pollard, D. (1984).Convergence of stochastic processes. Berlin: Springer.

## 256
## P. AUER, N. CESA-BIANCHI AND P. FISCHER
Sutton,  R.,  &  Barto,  A.  (1998).Reinforcement  learning,  an  introduction.  Cambridge:  MIT  Press/Bradford
## Books.
Wilks, S. (1962).Matematical statistics. New York: John Wiley and Sons.
Yakowitz,  S.,  &  Lowe,  W.  (1991).  Nonparametric  bandit  methods.Annals  of  Operations  Research,28,  297–
## 312.
## Received September 29, 2000
## Revised May 21, 2001
## Accepted June 20, 2001
Final manuscript June 20, 2001