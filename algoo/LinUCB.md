

arXiv:1003.0146v2  [cs.LG]  1 Mar 2012
A Contextual-Bandit Approach to
## Personalized News Article Recommendation
## Lihong Li
## †
## , Wei Chu
## †
## ,
## †
## Yahoo! Labs
lihong,chuwei@yahoo-
inc.com
## John Langford
## ‡
## ‡
## Yahoo! Labs
jl@yahoo-inc.com
## Robert E. Schapire
## +
## ∗
## +
Dept of Computer Science
## Princeton University
schapire@cs.princeton.edu
## ABSTRACT
Personalized web services strive to adapt their services (advertise-
ments,  news articles,  etc.)   to individual  users  by making  use  of
both content and user information.  Despite a few recent advances,
this problem remains challenging  for at least two reasons.   First,
web service is featured with dynamically changing pools of con-
tent,  rendering traditional collaborative filtering methods inappli-
cable.  Second, the scale of most web services of practical interest
calls for solutions that are both fast in learning and computation.
In this work,  we model personalized recommendation of news
articles as a contextual  bandit  problem,  a principled approach  in
which  a  learning  algorithm  sequentially  selects  articles  to  serve
users based on contextual information about the users and articles,
while simultaneously adapting  its article-selection strategy based
on user-click feedback to maximize total user clicks.
The contributions of this work are three-fold.  First, we propose
a new, general contextual bandit algorithm that is computationally
efficient and well motivated from learning theory.  Second, we ar-
gue that any bandit algorithm can be reliably evaluatedofflineus-
ing previously recorded random traffic.  Finally, using thisoffline
evaluation method, we successfully applied our new algorithm to
a  Yahoo!  Front  Page  Today  Module  dataset  containing  over33
million events.  Results showed a12.5%click lift compared to a
standard context-free bandit algorithm, and the advantagebecomes
even greater when data gets more scarce.
Categories and Subject Descriptors
H.3.5 [Information Systems]: On-line Information Services; I.2.6
[Computing Methodologies]: Learning
## General Terms
## Algorithms, Experimentation
## Keywords
Contextual bandit, web service, personalization, recommender sys-
tems, exploration/exploitation dilemma
## 1.INTRODUCTION
This paper addresses the challenge of identifying the most appro-
priate web-based content at the best time for individual users. Most
## ∗
This work was done while R. Schapire visited Yahoo! Labs.
A version of this paper appears at WWW 2010, April 26–30, 2010,
Raleigh, North Carolina, USA.
## .
service vendors acquire and maintain a large amount of content in
their repository, for instance, for filtering news articles[14] or for
the display of advertisements [5].  Moreover, the content ofsuch a
web-service repository changes dynamically, undergoing frequent
insertions and deletions.  In such a setting, it is crucial toquickly
identify interesting content  for  users.   For instance,  a news filter
must promptly identify the popularity of breaking news, while also
adapting to the fading value of existing, aging news stories.
It is generally difficult to model popularity and temporal changes
based solely on content information.   In practice,  we usually ex-
plore the unknown by collecting consumers’ feedback in realtime
to evaluate the popularity of new content while monitoring changes
in its value [3].  For instance, a small amount of traffic can bedes-
ignated for such exploration.  Based on the users’ response (such
as clicks) to randomly selected content on this small slice of traf-
fic, the most popular content can be identified and exploited on the
remaining traffic.  This strategy, with random exploration on anǫ
fraction of the traffic and greedy exploitation on the rest, is known
asǫ-greedy.  Advanced exploration approaches such asEXP3[8]
orUCB1[7] could be applied as well.  Intuitively, we need to dis-
tribute more traffic to new content to learn its value more quickly,
and fewer users to track temporal changes of existing content.
Recently, personalized recommendation has become a desirable
feature for websites to improve user satisfaction by tailoring con-
tent  presentation  to  suit  individual  users’  needs  [10].   Personal-
ization involves a process of gathering and storing user attributes,
managing content assets, and, based on an analysis of current and
past users’ behavior, delivering the individually best content to the
present user being served.
Often,  both  users  and  content  are  represented  by  sets  of  fea-
tures.  User features may include historical activities at an aggre-
gated level as well as declared demographic information.  Content
features may contain descriptive information and categories. In this
scenario, exploration and exploitation have to be deployedat an in-
dividual level since the views of different users on the samecon-
tent can vary significantly. Since there may be a very large number
of possible choices or actions available, it becomes critical to rec-
ognize commonalities between content  items and to transferthat
knowledge across the content pool.
Traditional  recommender  systems,  including  collaborative  fil-
tering, content-based filtering and hybrid approaches, canprovide
meaningful recommendations at an individual level by leveraging
users’ interests as demonstrated by their past activity. Collaborative
filtering [25], by recognizing similarities across users based on their
consumption history, provides a good recommendation solution to
the scenarios where overlap in historical consumption across users
is relatively high and the content universe is almost static. Content-
based  filtering  helps  to  identify  new  items  which  well  match  an

existing  user’s  consumption  profile,  but  the  recommended  items
are always similar to the items previously taken by the user [20].
Hybrid  approaches  [11]  have  been  developed  by  combining  two
or more recommendation techniques; for example, the inability of
collaborative filtering to recommend new items is commonly alle-
viated by combining it with content-based filtering.
However, as noted above, in many web-based scenarios, the con-
tent  universe  undergoes  frequent  changes,  with  content  popular-
ity changing  over  time as well.   Furthermore,  a significant  num-
ber of visitors are likely to be entirely new with no historical con-
sumption  record  whatsoever;  this is known  as acold-startsitua-
tion [21].  These issues make traditional recommender-system ap-
proaches difficult to apply, as shown by prior empirical studies [12].
It thus becomes indispensable to learn the goodness of matchbe-
tween user interests and content when one or both of them are new.
However,  acquiring  such  information  can  be  expensive  and  may
reduce user satisfaction in the short term,  raising the question of
optimally balancing the two competing goals: maximizing user sat-
isfaction in the long run, and gathering information about goodness
of match between user interests and content.
The above problem is indeed known as a feature-based explo-
ration/exploitation problem. In this paper, we formulate it as acon-
textual banditproblem, a principled approach in which a learning
algorithm sequentially selects articles to serve users based on con-
textual information of the user and articles, while simultaneously
adapting its article-selection strategy based on user-click feedback
to maximize total user clicks in the long run.  We define a bandit
problem and then review some existing approaches in Section2.
Then, we propose a new algorithm,LinUCB, in Section 3 which
has a similar regret analysis to the best known algorithms for com-
peting with the best linear predictor,  with a lower computational
overhead.   We  also  address  the  problem  ofofflineevaluation  in
Section  4,  showing  this  is possible  foranyexplore/exploit  strat-
egy when interactions are independent  and identically distributed
(i.i.d.), as might be a reasonable assumption for differentusers. We
then test our new algorithm and several existing algorithmsusing
this offline evaluation strategy in Section 5.
## 2.FORMULATION & RELATED WORK
In this section, we define theK-armed contextual bandit prob-
lem formally, and as an example, show how it can model the per-
sonalized news article recommendation problem.  We then discuss
existing methods and their limitations.
## 2.1    A Multi-armed Bandit Formulation
The problem of personalized news article recommendation can
be naturally modeled as a multi-armed bandit problem with context
information.  Following previous work [18], we call it acontextual
bandit.
## 1
Formally, a contextual-bandit algorithmAproceeds in dis-
crete trialst= 1,2,3,...In trialt:
-  The algorithm observes the current useru
t
and a setA
t
of
arms or actions together with their feature vectorsx
t,a
for
a∈A
t
. The vectorx
t,a
summarizes information ofboththe
useru
t
and arma, and will be referred to as thecontext.
-  Based on observed payoffs in previous trials,Achooses an
arma
t
## ∈ A
t
, and receives payoffr
t,a
t
whose expectation
depends on both the useru
t
and the arma
t
## .
-  The algorithm then improves its arm-selection strategy with
the new observation,(x
t,a
t
## ,a
t
## ,r
t,a
t
). It is important to em-
## 1
In the literature, contextual bandits are sometimes calledbandits
with covariate, bandits with side information, associative bandits,
and associative reinforcement learning.
phasize  here  thatnofeedback  (namely,  the  payoffr
t,a
)  is
observed forunchosenarmsa6=a
t
.  The consequence  of
this fact is discussed in more details in the next subsection.
In the process above, thetotalT-trial payoffofAis defined as
## ∑
## T
t=1
r
t,a
t
.Similarly, we define theoptimal expectedT-trial pay-
offasE
## [
## ∑
## T
t=1
r
t,a
## ∗
t
## ]
## ,wherea
## ∗
t
is the arm with maximum ex-
pected payoff at trialt. Our goal is to designAso that the expected
total payoff above is maximized.  Equivalently, we may find anal-
gorithm so that itsregretwith respect to the optimal arm-selection
strategy is minimized. Here, theT-trial regretR
## A
(T)of algorithm
Ais defined formally by
## R
## A
## (T)
def
## =E
## [
## T
## ∑
t=1
r
t,a
## ∗
t
## ]
## −E
## [
## T
## ∑
t=1
r
t,a
t
## ]
## .(1)
An important special case of the general contextual bandit prob-
lem is the well-knownK-armed banditin which (i) the arm setA
t
remains unchanged and containsKarms for allt, and (ii) the user
u
t
(or equivalently,  the context(x
t,1
## ,···,x
t,K
)) is the same for
allt. Since both the arm set and contexts are constant at every trial,
they make no difference to a bandit algorithm, and so we will also
refer to this type of bandit as acontext-freebandit.
In the context of article recommendation, we may view articles
in the pool as arms.  When a presented article is clicked, a payoff
of1is incurred;  otherwise,  the payoff  is0.   With this definition
of payoff,  the expected payoff of an article is precisely itsclick-
through rate (CTR), and choosing an article with maximum CTR
is equivalent  to maximizing  the expected  number  of  clicks from
users, which in turn is the same as maximizing the total expected
payoff in our bandit formulation.
Furthermore, in web services we often have access to user infor-
mation which can be used to infer a user’s interest and to choose
news articles that are probably most interesting to her. Forexample,
it is much more likely for a male teenager to be interested in an arti-
cle about iPod products rather than retirement plans. Therefore, we
may “summarize” users and articles by a set of informative features
that describe them compactly. By doing so, a bandit algorithm can
generalizeCTR information from one article/user to another, and
learn to choose good articles more quickly, especially for new users
and articles.
## 2.2    Existing Bandit Algorithms
The fundamental  challenge  in bandit  problems  is the need  for
balancing exploration and exploitation.  To minimize the regret in
Eq. (1), an algorithmAexploitsits past experience to select the arm
that appears best.  On the other hand, this seemingly optimalarm
may in fact be suboptimal, due to imprecision inA’s knowledge. In
order to avoid this undesired situation,Ahas toexploreby actually
choosing seemingly suboptimal arms so as to gather more informa-
tion about them (c.f., step 3 in the bandit process defined in the pre-
vious subsection). Exploration can increaseshort-termregret since
some suboptimal arms may be chosen.  However, obtaining infor-
mation about the arms’ average payoffs (i.e., exploration) can re-
fineA’s estimate of the arms’ payoffs and in turn reducelong-term
regret.  Clearly, neither a purely exploring nor a purely exploiting
algorithm works best in general, and a good tradeoff is needed.
The context-freeK-armed bandit problem has been studied by
statisticians for a long time [9,  24, 26].  One of the simplestand
most straightforward algorithms isǫ-greedy.  In each trialt, this
algorithm  first  estimates  the  average  payoffˆμ
t,a
of  each  arma.
Then, with probability1−ǫ, it chooses thegreedyarm (i.e., the
arm with highest payoff estimate); with probabilityǫ, it chooses a
random arm.  In the limit, each arm will be tried infinitely often,

and so the payoff estimateˆμ
t,a
converges to the true valueμ
a
with
probability1. Furthermore, by decayingǫappropriately (e.g., [24]),
the per-step regret,R
## A
(T)/T, converges to0with probability1.
In contrast  to theunguidedexploration  strategy adopted  byǫ-
greedy, another class of algorithms generally known as upper con-
fidence bound algorithms [4, 7, 17] use a smarter way to balance
exploration  and  exploitation.   Specifically,  in  trialt,  these  algo-
rithms estimate both the mean payoffˆμ
t,a
of each armaas well
as a corresponding confidence intervalc
t,a
, so that|ˆμ
t,a
## −μ
a
## |<
c
t,a
holds  with  high  probability.   They  then  select  the  arm  that
achieves a highest upper confidence bound (UCB for short):a
t
## =
arg max
a
## (ˆμ
t,a
## +c
t,a
). With appropriately defined confidence in-
tervals, it can be shown that such algorithms have a small totalT-
trial regret that is only logarithmic in the total number of trialsT,
which turns out to be optimal [17].
While context-freeK-armed bandits are extensively studied and
well understood,  the more general contextual bandit problem has
remained challenging.  TheEXP4algorithm [8] uses the exponen-
tial weighting technique to achieve an
## ̃
## O(
## √
## T)regret,
## 2
but the com-
putational  complexity  may  be  exponential  in  the number  of  fea-
tures.  Another general contextual bandit algorithm is theepoch-
greedyalgorithm [18] that is similar toǫ-greedywith shrinking
ǫ. This algorithm is computationally efficient given an oracle opti-
mizer but has the weaker regret guarantee of
## ̃
## O(T
## 2/3
## ).
Algorithms with stronger regret guarantees may be designedun-
der various modeling assumptions about the bandit.  Assuming the
expected  payoff  of  an  arm  is  linear  in  its features,  Auer  [6]  de-
scribes  theLinRelalgorithm  that  is  essentially  a  UCB-type  ap-
proach and shows that one of its variants has a regret of
## ̃
## O(
## √
T), a
significant improvement over earlier algorithms [1].
Finally,  we  note  that  there  exist  another  class  of  bandit  al-
gorithms   based   on  Bayes   rule,   such  as   Gittins  index   meth-
ods [15].  With appropriately defined prior distributions, Bayesian
approaches may have good performance.  These methods require
extensive offline engineering to obtain good prior models, and are
often computationally prohibitive without coupling with approxi-
mation techniques [2].
## 3.ALGORITHM
Given asymptotic optimality and the strong regret bound of UCB
methods  for  context-free  bandit  algorithms,  it  is temptingto de-
vise similar algorithms for contextual bandit problems. Given some
parametric form of payoff function, a number of methods exist to
estimate from data the confidence interval of the parameterswith
which we can compute a UCB of the estimated arm payoff.  Such
an approach, however, is expensive in general.
In this work,  we show  that  a confidence  interval  can  be com-
putedefficiently in closed  formwhen  the payoff  model  is linear,
and call this algorithmLinUCB. For convenience of exposition, we
first describe the simpler form fordisjointlinear models, and then
consider the general case ofhybridmodels in Section 3.2. We note
LinUCBis a generic contextual bandit algorithms which applies to
applications other than personalized news article recommendation.
3.1    LinUCB with Disjoint Linear Models
Using the notation of Section 2.1, we assume the expected payoff
of an armais linear in itsd-dimensional featurex
t,a
with some
unknown coefficient vectorθ
θ
θ
## ∗
a
; namely, for allt,
## E[r
t,a
## |x
t,a
## ]  =x
## ⊤
t,a
θ
θ
θ
## ∗
a
## .(2)
This model is calleddisjointsince the parameters are not shared
## 2
## Note
## ̃
O(·)is the same asO(·)but suppresses logarithmic factors.
among  different  arms.   LetD
a
be a design  matrix of dimension
m×dat trialt, whose rows correspond tomtraining inputs (e.g.,
mcontexts that are observed previously for articlea), andb
a
## ∈
## R
m
be the corresponding response vector (e.g., the corresponding
mclick/no-click user feedback).  Applying ridge regressionto the
training data(D
a
## ,c
a
)gives an estimate of the coefficients:
## ˆ
θ
θ
θ
a
## = (D
## ⊤
a
## D
a
## +I
d
## )
## −1
## D
## ⊤
a
c
a
## ,(3)
whereI
d
is thed×didentity matrix. When components inc
a
are
independent  conditioned on corresponding rows inD
a
, it can be
shown [27] that, with probability at least1−δ,
## ∣
## ∣
## ∣
x
## ⊤
t,a
## ˆ
θ
θ
θ
a
−E[r
t,a
## |x
t,a
## ]
## ∣
## ∣
## ∣
## ≤α
## √
x
## ⊤
t,a
## (D
## ⊤
a
## D
a
## +I
d
## )
## −1
x
t,a
## (4)
for anyδ >0andx
t,a
## ∈R
d
, whereα= 1 +
## √
ln(2/δ)/2is a
constant.  In other words, the inequality above gives a reasonably
tight UCB for the expected payoff of arma, from which a UCB-
type arm-selection strategy can be derived: at each trialt, choose
a
t
def
= arg max
a∈A
t
## (
x
## ⊤
t,a
## ˆ
θ
θ
θ
a
## +α
## √
x
## ⊤
t,a
## A
## −1
a
x
t,a
## )
## ,(5)
whereA
a
def
## =D
## ⊤
a
## D
a
## +I
d
## .
The confidence interval in Eq. (4) may be motivated and derived
from other principles.   For instance,  ridge regression can also be
interpreted as a Bayesian point estimate, where the posterior dis-
tribution of the coefficient  vector,  denoted  asp(θ
θ
θ
a
),  is Gaussian
with mean
## ˆ
θ
θ
θ
a
and covarianceA
## −1
a
.  Given the current model, the
predictive variance of the expected payoffx
## ⊤
t,a
θ
θ
θ
## ∗
a
is evaluated as
x
## ⊤
t,a
## A
## −1
a
x
t,a
, and then
## √
x
## ⊤
t,a
## A
## −1
a
x
t,a
becomes the standard de-
viation.  Furthermore,  in information theory [19],  the differential
entropy ofp(θ
θ
θ
a
)is defined as−
## 1
## 2
ln((2π)
d
detA
a
).  The entropy
ofp(θ
θ
θ
a
)when updated by the inclusion of the new pointx
t,a
then
becomes−
## 1
## 2
ln((2π)
d
det (A
a
## +x
t,a
x
## ⊤
t,a
)).  The entropy reduc-
tion in the model posterior is
## 1
## 2
ln(1 +x
## ⊤
t,a
## A
## −1
a
x
t,a
).  This quan-
tity is often used to evaluate model improvement contributed from
x
t,a
.  Therefore, the criterion for arm selection in Eq. (5) can also
be regarded  as an additive  trade-off  between  the payoff  estimate
and model uncertainty reduction.
Algorithm 1 gives a detailed description of the entireLinUCB
algorithm, whose only input parameter isα.  Note the value ofα
given in Eq. (4) may be conservatively large in some applications,
and so optimizing this parameter may result in higher total payoffs
in practice.  Like all UCB methods,LinUCBalways chooses the
arm with highest UCB (as in Eq. (5)).
This algorithm has a few nice properties. First, its computational
complexity is linear in the number  of  arms and at most  cubic in
the number of features.  To decrease computation further, wemay
updateA
a
t
in every step (which takesO(d
## 2
)time), but compute
and  cacheQ
a
def
## =A
## −1
a
(for  alla)  periodically  instead  of  in  real-
time.   Second,  the  algorithm  works  well  for  a  dynamic  arm  set,
and remains efficient as long as the size ofA
t
is not too large. This
case is true in many applications. In news article recommendation,
for instance, editors add/remove articles to/from a pool and the pool
size remains essentially constant. Third, although it is not the focus
of the present paper, we can adapt the analysis from [6] to show the
following: if the arm setA
t
is fixed and containsKarms, then the
confidence interval (i.e., the right-hand side of Eq. (4)) decreases
fast enough with more and more data,  and then prove the strong
regret bound of
## ̃
## O(
## √
KdT), matching the state-of-the-art result [6]
for  bandits  satisfying  Eq.  (2).   These  theoretical  results  indicate
fundamental soundness and efficiency of the algorithm.

Algorithm 1LinUCB with disjoint linear models.
0:  Inputs:α∈R
## +
1:fort= 1,2,3,...,Tdo
2:Observe features of all armsa∈A
t
## :x
t,a
## ∈R
d
3:for alla∈A
t
do
4:ifais newthen
## 5:A
a
## ←I
d
(d-dimensional identity matrix)
## 6:b
a
## ←0
d×1
(d-dimensional zero vector)
7:end if
## 8:
## ˆ
θ
θ
θ
a
## ←A
## −1
a
b
a
## 9:p
t,a
## ←
## ˆ
θ
θ
θ
## ⊤
a
x
t,a
## +α
## √
x
## ⊤
t,a
## A
## −1
a
x
t,a
10:end for
11:Choose arma
t
= arg max
a∈A
t
p
t,a
with ties broken arbi-
trarily, and observe a real-valued payoffr
t
## 12:A
a
t
## ←A
a
t
## +x
t,a
t
x
## ⊤
t,a
t
## 13:b
a
t
## ←b
a
t
## +r
t
x
t,a
t
14:end for
Finally,  we note that,  under the assumption that input features
x
t,a
were drawn i.i.d. from a normal distribution (in addition tothe
modeling assumption in Eq. (2)), Pavlidiset al.[22] came up with
a similar algorithm that uses a least-squares solution
## ̃
θ
θ
θ
a
instead of
our ridge-regression solution (
## ˆ
θ
θ
θ
a
in Eq. (3)) to compute the UCB.
However, our approach (and theoretical analysis) is more general
and remains valid even when input features are nonstationary. More
importantly, we will discuss in the next section how to extend the
basic Algorithm 1 to a much more interesting case not coveredby
Pavlidiset al.
3.2    LinUCB with Hybrid Linear Models
Algorithm 1 (or the similar algorithm in [22]) computes the in-
verse of the matrix,D
## ⊤
a
## D
a
## +I
d
(orD
## ⊤
a
## D
a
), whereD
a
is again
the design matrix with rows corresponding to features in thetrain-
ing data.  These matrices of all arms have fixed dimensiond×d,
and can be updated efficiently and incrementally.  Moreover,their
inverses can be computed easily as the parameters in Algorithm 1
aredisjoint:  the solution
## ˆ
θ
θ
θ
a
in Eq. (3) is not affected by training
data of other arms, and so can be computed separately.  We now
consider the more interesting case withhybridmodels.
In many applications including ours, it is helpful to use features
that are shared by all arms, in addition to the arm-specific ones. For
example, in news article recommendation, a user may prefer only
articles about politics for which this provides a mechanism. Hence,
it is helpful to have features that have both shared and non-shared
components.   Formally,  we adopt  the followinghybrid modelby
adding another linear term to the right-hand side of Eq. (2):
## E[r
t,a
## |x
t,a
## ]  =z
## ⊤
t,a
β
β
β
## ∗
## +x
## ⊤
t,a
θ
θ
θ
## ∗
a
## ,(6)
wherez
t,a
## ∈R
k
is the feature of the current user/article combina-
tion, andβ
β
β
## ∗
is an unknown coefficient vector common to all arms.
This model is hybrid in the sense that some of the coefficientsβ
β
β
## ∗
are shared by all arms, while othersθ
θ
θ
## ∗
a
are not.
For  hybrid  models,  we  can  no  longer  use  Algorithm  1  as  the
confidence intervals of various arms are not independent dueto the
shared features.  Fortunately, there is an efficient way to compute
an UCB along the same line of reasoning as in the previous sec-
tion.  The derivation relies heavily on block matrix inversion tech-
niques.  Due to space limitation, we only give the pseudocodein
Algorithm 2 (where lines 5 and 12 compute the ridge-regression
solution of the coefficients,  and line 13 computes the confidence
interval), and leave detailed derivations to a full paper.  Here, we
Algorithm 2LinUCB with hybrid linear models.
0:  Inputs:α∈R
## +
## 1:A
## 0
## ←I
k
(k-dimensional identity matrix)
## 2:b
## 0
## ←0
k
(k-dimensional zero vector)
3:fort= 1,2,3,...,Tdo
4:Observe features of all armsa∈A
t
## :(z
t,a
## ,x
t,a
## )∈R
k+d
## 5:
## ˆ
β
β
β←A
## −1
## 0
b
## 0
6:for alla∈A
t
do
7:ifais newthen
## 8:A
a
## ←I
d
(d-dimensional identity matrix)
## 9:B
a
## ←0
d×k
(d-by-kzero matrix)
## 10:b
a
## ←0
d×1
(d-dimensional zero vector)
11:end if
## 12:
## ˆ
θ
θ
θ
a
## ←A
## −1
a
## (
b
a
## −B
a
## ˆ
β
β
β
## )
## 13:s
t,a
## ←z
## ⊤
t,a
## A
## −1
## 0
z
t,a
## −2z
## ⊤
t,a
## A
## −1
## 0
## B
## ⊤
a
## A
## −1
a
x
t,a
## +
x
## ⊤
t,a
## A
## −1
a
x
t,a
## +x
## ⊤
t,a
## A
## −1
a
## B
a
## A
## −1
## 0
## B
## ⊤
a
## A
## −1
a
x
t,a
## 14:p
t,a
## ←z
## ⊤
t,a
## ˆ
β
β
β+x
## ⊤
t,a
## ˆ
θ
θ
θ
a
## +α
## √
s
t,a
15:end for
16:Choose arma
t
= arg max
a∈A
t
p
t,a
with ties broken arbi-
trarily, and observe a real-valued payoffr
t
## 17:A
## 0
## ←A
## 0
## +B
## ⊤
a
t
## A
## −1
a
t
## B
a
t
## 18:b
## 0
## ←b
## 0
## +B
## ⊤
a
t
## A
## −1
a
t
b
a
t
## 19:A
a
t
## ←A
a
t
## +x
t,a
t
x
## ⊤
t,a
t
## 20:B
a
t
## ←B
a
t
## +x
t,a
t
z
## ⊤
t,a
t
## 21:b
a
t
## ←b
a
t
## +r
t
x
t,a
t
## 22:A
## 0
## ←A
## 0
## +z
t,a
t
z
## ⊤
t,a
t
## −B
## ⊤
a
t
## A
## −1
a
t
## B
a
t
## 23:b
## 0
## ←b
## 0
## +r
t
z
t,a
t
## −B
## ⊤
a
t
## A
## −1
a
t
b
a
t
24:end for
only point out the important fact that the algorithm is computation-
ally efficient since the building blocks in the algorithm (A
## 0
## ,b
## 0
## ,
## A
a
## ,B
a
,  andb
a
)  all have fixed  dimensions  and can  be updated
incrementally.   Furthermore,  quantities  associated  with  arms  not
existing inA
t
no longer get involved in the computation.  Finally,
we can also compute and cache the inverses (A
## −1
## 0
andA
## −1
a
) pe-
riodically instead of at the end of each trial to reduce the per-trial
computational complexity toO(d
## 2
## +k
## 2
## ).
## 4.    EVALUATION METHODOLOGY
Compared to machine learning in the more standard supervised
setting, evaluation of methods in a contextual bandit setting is frus-
tratingly difficult. Our goal here is to measure the performance of a
bandit algorithmπ, that is, a rule for selecting an arm at each time
step based on the preceding interactions (such as the algorithms de-
scribed above). Because of the interactive nature of the problem, it
would seem that the only way to do this is to actually run the algo-
rithm on “live” data. However, in practice, this approach islikely to
be infeasible due to the serious logistical challenges thatit presents.
Rather, we may only haveofflinedata available that was collected
at a previous time using an entirelydifferentlogging policy.  Be-
cause payoffs are only observed for the arms chosen by the logging
policy,  which are likely to often differ from those chosen  bythe
algorithmπbeing evaluated,  it is not at all clear how to evaluate
πbased only on such logged data.  This evaluation problem may
be viewed as a special case of the so-called “off-policy evaluation
problem” in reinforcement learning (see,c.f., [23]).
One solution is to build a simulator to model the bandit process
from the logged data, and then evaluateπwith the simulator. How-
ever, the modeling step will introducebiasin the simulator and so
make it hard to justify the reliability of this simulator-based evalu-

ation approach.  In contrast, we propose an approach that is simple
to implement, grounded on logged data, andunbiased.
In this section, we describe a provably reliable technique for car-
rying out such an evaluation,  assuming that the individual events
are i.i.d.,  and that the logging policy that was used to gather the
logged data chose each arm at each time step uniformly at random.
Although we omit the details, this latter assumption can be weak-
ened considerably so that any randomized logging policy is allowed
and our solution can be modified accordingly using rejectionsam-
pling, but at the cost of decreased efficiency in using data.
More  precisely,  we  suppose  that  there  is  some  unknown  dis-
tributionDfrom  which  tuples  are  drawn  i.i.d.  of  the  form
## (x
## 1
## ,...,x
## K
## ,r
## 1
## ,...,r
## K
), each consisting of observed feature vec-
tors andhiddenpayoffs for all arms. We also posit access to a large
sequence of logged events resulting from the interaction ofthe log-
ging policy with the world. Each such event consists of the context
vectorsx
## 1
## ,...,x
## K
, a selected armaand the resulting observed pay-
offr
a
. Crucially, only the payoffr
a
is observed for the single arm
athat was chosen uniformly at random. For simplicity of presenta-
tion, we take this sequence of logged events to be an infinitely long
stream; however, we also give explicit bounds on the actual finite
number of events required by our evaluation method.
Our  goal  is  to  use  this  data  to  evaluate  a  bandit  algorithmπ.
Formally,πis a (possibly randomized) mapping for selecting the
arma
t
at timetbased on the historyh
t−1
oft−1preceding events,
together with the current context vectorsx
t1
## ,...,x
tK
## .
Our  proposed  policy  evaluator  is  shown  in  Algorithm  3.   The
method takes as input a policyπand a desired number of “good”
eventsTon which to base the evaluation.  We then step through
the stream of logged events one by one.  If, given the current his-
toryh
t−1
, it happens that the policyπchooses the same armaas
the one that was selected by the logging policy, then the event is
retained, that is, added to the history, and the total payoffR
t
up-
dated.  Otherwise, if the policyπselects a different arm from the
one that was taken by the logging policy, then the event is entirely
ignored, and the algorithm proceeds to the next event without any
other change in its state.
Note  that,  because  the  logging  policy  chooses  each  arm  uni-
formly  at  random,  each  event  is  retained  by  this  algorithm  with
probability  exactly1/K,  independent  of  everything  else.    This
means that the events which are retained have the same distribution
as if they were selected byD.  As a result, we can prove that two
processes are equivalent: the first is evaluating the policyagainstT
real-world events fromD, and the second is evaluating the policy
using the policy evaluator on a stream of logged events.
THEOREM1.For all distributionsDof contexts, all policiesπ,
allT, and all sequences of eventsh
## T
## ,
## Pr
Policy_Evaluator(π,S)
## (h
## T
## ) = Pr
π,D
## (h
## T
## )
whereSis a stream of events drawn i.i.d.  from a uniform random
logging policy andD. Furthermore, the expected number of events
obtained from the stream to gather a historyh
## T
of lengthTisKT.
This theorem says thateveryhistoryh
## T
has the identical prob-
ability in the real world as in the policy evaluator.  Many statistics
of these histories,  such as the average payoffR
## T
/Treturned by
Algorithm 3, are therefore unbiased estimates of the value of the
algorithmπ. Further, the theorem states thatKTlogged events are
required, in expectation, to retain a sample of sizeT.
PROOF.  The proof is by induction ont= 1,...,Tstarting with
a base case of the empty history which has probability1whent= 0
Algorithm 3Policy_Evaluator.
0:  Inputs:T >0; policyπ; stream of events
## 1:h
## 0
←∅{An initially empty history}
## 2:R
## 0
←0{An initially zero total payoff}
3:fort= 1,2,3,...,Tdo
## 4:repeat
5:Get next event(x
## 1
## ,...,x
## K
## ,a,r
a
## )
## 6:untilπ(h
t−1
## ,(x
## 1
## ,...,x
## K
## )) =a
## 7:h
t
←CONCATENATE(h
t−1
## ,(x
## 1
## ,...,x
## K
## ,a,r
a
## ))
## 8:R
t
## ←R
t−1
## +r
a
9:end for
10:  Output:R
## T
## /T
under both methods of evaluation.  In the inductive case,  assume
that we have for allt−1:
## Pr
Policy_Evaluator(π,S)
## (h
t−1
## ) = Pr
π,D
## (h
t−1
## )
and want to prove the same statement for any historyh
t
. Since the
data is i.i.d. and any randomization in the policy is independent of
randomization in the world, we need only prove that conditioned
on the historyh
t−1
the distribution over thet-th event is the same
for each process. In other words, we must show:
## Pr
Policy_Evaluator(π,S)
## ((x
t,1
## ,...,x
t,K
## ,a,r
t,a
## )|h
t−1
## )
## = Pr
## D
## (x
t,1
## ,...,x
t,K
## ,r
t,a
## )  Pr
π(h
t−1
## )
## (a|x
t,1
## ,...,x
t,K
## ).
Since the armais chosen uniformly at random in the logging pol-
icy, the probability that the policy evaluator exits the inner loop is
identical for any policy, any history, any features, and anyarm, im-
plying  this happens  for  the last event  with the probability  of the
last event,Pr
## D
## (x
t,1
## ,...,x
t,K
## ,r
t,a
). Similarly, since the policyπ’s
distribution  over  arms  is  independent  conditioned  on  the  history
h
t−1
and features(x
t,1
## ,...,x
t,K
), the probability of armais just
## Pr
π(h
t−1
## )
## (a|x
t,1
## ,...,x
t,K
## ).
Finally, since each event from the stream is retained with proba-
bility exactly1/K, the expected number required to retainTevents
is exactlyKT.
## 5.    EXPERIMENTS
In this section, we verify the capacity of the proposedLinUCB
algorithm on a real-world application using the offline evaluation
method of Section 4. We start with an introduction of the problem
setting in Yahoo! Today-Module,  and then describe the user/item
attributes we used in experiments.  Finally, we define performance
metrics and report experimental results with comparison toa few
standard (contextual) bandit algorithms.
## 5.1    Yahoo! Today Module
The Today Module is the most prominent panel on the Yahoo!
Front Page, which is also one of the most visited pages on the In-
ternet; see a snapshot in Figure 1. The default “Featured” tab in the
Today Module highlights one of four high-quality articles,mainly
news, while the four articles are selected from an hourly-refreshed
article pool curated by human editors.  As illustrated in Figure 1,
there are four articles at footer positions, indexed by F1–F4.  Each
article is represented by a small picture and a title. One of the four
articles is highlighted at the story position, which is featured by a
large picture, a title and a short summary along with relatedlinks.
By default, the article at F1 is highlighted at the story position.  A

Figure 1: A snapshot of the “Featured” tab in the Today Mod-
ule on Yahoo! Front Page. By default, the article at F1 position
is highlighted at the story position.
user can click on the highlighted article at the story position to read
more details if she is interested in the article. The event isrecorded
as a story click.  To draw visitors’ attention, we would like to rank
available articles according to individual interests, andhighlight the
most attractive article for each visitor at the story position.
## 5.2    Experiment Setup
This subsection gives a detailed description of our experimental
setup, including data collection, feature construction, performance
evaluation, and competing algorithms.
## 5.2.1    Data Collection
We collected events from a random bucket in May 2009.  Users
were randomly selected to the bucket with a certain probability per
visiting view.
## 3
In this bucket, articles were randomly selected from
the  article pool  to serve  users.   To avoid exposure  bias at  footer
positions, we only focused on users’ interactions with F1 articles
at the story position.  Each user interactioneventconsists of three
components:  (i) the random  article chosen  to serve the user,(ii)
user/article information, and (iii) whether the user clicks on the ar-
ticle at the story position. Section 4 shows these random events can
be used to reliably evaluate a bandit algorithm’s expected payoff.
There were  about4.7million events  in the  random  bucket  on
May 01. We used this day’s events (called “tuning data”) for model
validation to decide the optimal parameter for each competing ban-
dit algorithm. Then we ran these algorithms with tuned parameters
on a one-week event set (called “evaluation data”) in the random
bucket from May 03–09, which contained about36million events.
## 5.2.2    Feature Construction
We now describe the user/article features constructed for our ex-
periments. Two sets of features for the disjoint and hybrid models,
respectively,  were used to test the two forms ofLinUCBin Sec-
tion 3 and to verify our conjecture that hybrid models can improve
learning speed.
We start with raw user features that were selected by “support”.
The support of a feature is the fraction of users having that feature.
To reduce noise in the data,  we only selected features with high
support. Specifically, we used a feature when its support is at least
0.1.  Then, each user was originally represented by a raw feature
vector of over1000categorical components, which include: (i) de-
mographic information: gender (2classes) and age discretized into
10segments; (ii) geographic features:  about200metropolitan lo-
cations worldwide and U.S. states; and (iii) behavioral categories:
## 3
We  call  it  view-based  randomization.After  refreshing  her
browser, the user may not fall into the random bucket again.
about1000binary categories that summarize the user’s consump-
tion history within Yahoo! properties. Other than these features, no
other information was used to identify a user.
Similarly, each article was represented by a raw feature vector of
about100categorical features constructed in the same way. These
features include:  (i) URL categories:  tens of classes inferred from
the URL of the article resource; and (ii) editor categories:tens of
topics tagged by human editors to summarize the article content.
We  followed  a  previous  procedure  [12]  to  encode  categorical
user/article features as binary vectors and then normalizeeach fea-
ture vector to unit length.  We also augmented each feature vector
with a constant feature of value1.  Now each article and user was
represented by a feature vector of83and1193entries, respectively.
To  further  reduce  dimensionality  and  capture  nonlinearityin
these raw features, we carried out conjoint analysis based on ran-
dom  exploration data collected  in September  2008.   Following a
previous approach to dimensionality reduction [13], we projected
user features onto article categories and then clustered users with
similar preferences into groups. More specifically:
•We first used logistic regression (LR) to fit a bilinear model
for  click probability  given  raw user/article  features so  that
φ
φ
φ
## ⊤
u
## Wφ
φ
φ
a
approximated the probability that the useruclicks
on articlea, whereφ
φ
φ
u
andφ
φ
φ
a
were the corresponding feature
vectors, andWwas a weight matrix optimized by LR.
•Raw user features were then projected onto an induced space
by computingψ
ψ
ψ
u
def
## =φ
φ
φ
## ⊤
u
W.  Here, thei
th
component inψ
ψ
ψ
u
for userumay be interpreted as the degree to which the user
likes thei
th
category  of  articles.   K-means was applied to
group users in the inducedψ
ψ
ψ
u
space into5clusters.
•The  final  user  feature  was  a  six-vector:  five  entries corre-
sponded to membership of that user in these5clusters (com-
puted  with  a  Gaussian  kernel  and  then  normalized  so  that
they sum up to unity), and the sixth was a constant feature1.
At trialt, each articleahas a separate six-dimensional featurex
t,a
that is exactly the six-dimensional feature constructed asabove for
useru
t
.   Since these article features do not  overlap,  they are for
disjoint linear models defined in Section 3.
For each articlea, we performed the same dimensionality reduc-
tion to obtain a six-dimensional article feature (including a constant
1feature).  Its outer product with a user feature gave6×6 = 36
features, denotedz
t,a
## ∈R
## 36
, that corresponded to the shared fea-
tures in Eq. (6), and thus(z
t,a
## ,x
t,a
)could be used in the hybrid
linear model.  Note the featuresz
t,a
contains user-article interac-
tion information, whilex
t,a
contains user information only.
Here,  we  intentionally  used  five  users  (and  articles)  groups,
which has been shown to be representative in segmentation anal-
ysis [13]. Another reason for using a relatively small feature space
is that, in online services, storing and retrieving large amounts of
user/article information will be too expensive to be practical.
## 5.3    Compared Algorithms
The algorithms empirically evaluated in our experiments can be
categorized into three groups:
I. Algorithms that make no use of features.These correspond to
the context-freeK-armed bandit algorithms that ignore all contexts
(i.e., user/article information).
•random: A random policy always chooses one of the candi-
date articles from the pool with equal probability. This algo-
rithm requires no parameters and does not “learn” over time.
•ǫ-greedy: As described in Section 2.2, it estimates each arti-
cle’s CTR; then it chooses a random article with probability
ǫ, and chooses the article of the highest CTR estimate with
probability1−ǫ. The only parameter of this policy isǫ.

## 1
## 1.2
## 1.4
## 1.6
## 1.8
## 2
## 0 0.2 0.4 0.6 0.8 1
ctr
ε
ε-greedy
ε-greedy (warm)
ε-greedy (seg)
ε-greedy (disjoint)
ε-greedy (hybrid)
omniscient
(a) Deployment bucket.
## 1
## 1.2
## 1.4
## 1.6
## 1.8
## 2
## 0 0.2 0.4 0.6 0.8 1 1.2 1.4
ctr
α
ucb
ucb (warm)
ucb (seg)
linucb (disjoint)
linucb (hybrid)
omniscient
(b) Deployment bucket.
## 1
## 1.2
## 1.4
## 1.6
## 1.8
## 2
## 0 0.2 0.4 0.6 0.8 1
ctr
ε
ε-greedy
ε-greedy (warm)
ε-greedy (seg)
ε-greedy (disjoint)
ε-greedy (hybrid)
omniscient
(c) Learning bucket.
## 1
## 1.2
## 1.4
## 1.6
## 1.8
## 2
## 0 0.2 0.4 0.6 0.8 1 1.2 1.4
ctr
α
ucb
ucb (warm)
ucb (seg)
linucb (simple)
linucb (hybrid)
omniscient
(d) Learning bucket.
Figure 2: Parameter tuning: CTRs of various algorithms on the one-day tuning dataset.
•ucb: As described in Section 2.2, this policy estimates each
article’s CTR as well as a confidence interval of the estimate,
and always chooses the article with the highest UCB. Specifi-
cally, followingUCB1[7], we computed an articlea’s confi-
dence interval byc
t,a
## =
α
## √
n
t,a
, wheren
t,a
is the number of
timesawas chosen prior to trialt, andα >0is a parameter.
•omniscient:   Such  a  policy  achieves  the  best  empirical
context-free CTR fromhindsight.  It first computes each ar-
ticle’s empirical CTR from logged events, and then always
chooses  the  article  with  highest  empircal  CTR  when  it  is
evaluated using thesamelogged events.  This algorithm re-
quires no parameters and does not “learn” over time.
II. Algorithms with “warm start”—an intermediate step towards
personalized services.  The idea is to provide an offline-estimated
user-specific  adjustment  on  articles’  context-free  CTRs  over  the
whole traffic. The offset serves as an initialization on CTR estimate
for new content, a.k.a.“warm start”.  We re-trained the bilinear lo-
gistic regression model studied in [12] on Sept 2008 random traffic
data, using featuresz
t,a
constructed above. The selection criterion
then becomes the sum of the context-free CTR estimate and a bi-
linear term for a user-specific CTR adjustment.  In training,CTR
was estimated using the context-freeǫ-greedywithǫ= 1.
•ǫ-greedy (warm):  This algorithm is the same asǫ-greedy
except it adds the user-specific CTR correction to the article’s
context-free CTR estimate.
•ucb (warm): This algorithm is the same as the previous one
but replacesǫ-greedywithucb.
III. Algorithms that learn user-specific CTRs online.
•ǫ-greedy (seg):  Each  user  is  assigned  to  the  closest  user
cluster among the five constructed in Section 5.2.2, and so all
users are partitioned into five groups (a.k.a. user segments),
in each of which a separate copy ofǫ-greedywas run.
•ucb (seg):  This algorithm is similar toǫ-greedy (seg)ex-
cept it ran a copy ofucbin each of the five user segments.
•ǫ-greedy (disjoint): This isǫ-greedywith disjoint models,
and may be viewed as a close variant ofepoch-greedy[18].
•linucb (disjoint): This is Algorithm 1 with disjoint models.
•ǫ-greedy (hybrid):  This  isǫ-greedywith hybrid  models,
and may be viewed as a close variant ofepoch-greedy.
•linucb (hybrid): This is Algorithm 2 with hybrid models.
## 5.4    Performance Metric
An  algorithm’s  CTR  is  defined  as  the  ratio  of  the  number  of
clicks it receives and the number  of  steps it is run.   We used all
algorithms’  CTRs on the  random  logged  events  for  performance
comparison.  To protect business-sensitive information, we report
an algorithm’srelative CTR, which is the algorithm’s CTR divided
by the random policy’s. Therefore, we will not report a random pol-
icy’s relative CTR as it is always1by definition. For convenience,
we will use the term “CTR” from now on instead of “relative CTR”.
For  each  algorithm,  we  are  interested  in  two  CTRs motivated
by our application, which may be useful for other similar applica-
tions.   When deploying  the methods  to Yahoo!’s front page,  one
reasonable way is to randomly split all traffic to this page into two
buckets [3]. The first, called “learning bucket”, usually consists of
a small fraction of traffic on which various bandit algorithms are
run to learn/estimate article CTRs.  The other, called “deployment
bucket”,  is where Yahoo! Front Page greedily serves users using
CTR estimates obained from the learning bucket. Note that “learn-
ing” and “deployment” are interleaved in this problem, and so in
every view falling into the deployment bucket, the article with the
highestcurrent(user-specific) CTR estimate is chosen;  this esti-
mate may change later if the learning bucket gets more data. CTRs
in both buckets were estimated with Algorithm 3.

algorithm
size = 100%size = 30%size = 20%size = 10%size = 5%size = 1%
deploylearndeploylearndeploylearndeploylearndeploylearndeploylearn
ǫ-greedy
## 1.5961.3261.5411.3261.5491.2731.4651.3261.4091.2921.2341.139
## 0%0%0%0%0%0%0%0%0%0%0%0%
ucb
## 1.5941.5691.5821.5351.5691.4881.5411.4461.5411.4651.3541.22
## 0%18.3%2.7%15.8%1.3%16.9%5.2%9%9.4%13.4%9.7%7.1%
ǫ-greedy (seg)
## 1.7421.4461.6521.461.5851.1191.4741.2841.4071.2811.2451.072
## 9.1%9%7.2%10.1%2.3%−12%0.6%−3.1%0%−0.8%0.9%−5.8%
ucb (seg)
## 1.7811.6771.7421.5551.6891.4461.6361.5291.5321.321.3981.25
## 11.6%26.5%13%17.3%9%13.6%11.7%15.3%8.7%2.2%13.3%9.7%
ǫ-greedy (disjoint)
## 1.7691.3091.6861.3371.6241.5291.5291.4511.4321.3451.2621.183
## 10.8%−1.2%9.4%0.8%4.8%20.1%4.4%9.4%1.6%4.1%2.3%3.9%
linucb (disjoint)
## 1.7951.6471.7191.5071.7141.3841.6551.3871.5741.2451.3821.197
## 12.5%24.2%11.6%13.7%10.7%8.7%13%4.6%11.7%−3.5%12%5.1%
ǫ-greedy (hybrid)
## 1.7391.5211.681.3451.6361.4491.581.3481.4651.4151.3421.2
## 9%14.7%9%1.4%5.6%13.8%7.8%1.7%4%9.5%8.8%5.4%
linucb (hybrid)
## 1.731.6631.6911.5911.7081.6191.6751.5351.5881.5071.4821.446
## 8.4%25.4%9.7%20%10.3%27.2%14.3%15.8%12.7%16.6%20.1%27%
Table 1: Performance evaluation: CTRs of all algorithms on the one-week evaluation dataset in the deployment and learning buckets
(denoted by “deploy” and “learn” in the table, respectively). The numbers with a percentage is the CTR lift compared toǫ-greedy.
Since  the  deployment  bucket  is  often  larger  than  the  learning
bucket,  CTR in the deployment  bucket is more important.  How-
ever, a higher CTR in the learning bucket suggests a faster learning
rate (or equivalently, smaller regret) for a bandit algorithm. There-
fore, we chose to report algorithm CTRs in both buckets.
## 5.5    Experimental Results
5.5.1    Results for Tuning Data
Each of the competing algorithms (exceptrandomandomni-
scient) in Section 5.3 requires a single parameter:ǫforǫ-greedy
algorithms andαfor UCB ones.  We used tuning data to optimize
these parameters.  Figure 2 shows how the CTR of each algorithm
changes with respective parameters.  All results were obtained by
a single run, but given the size of our dataset and the unbiasedness
result in Theorem 1, the reported numbers are statisticallyreliable.
First, as seen from Figure 2, the CTR curves in the learning buck-
ets often possess the inverted U-shape.  When the parameter (ǫor
α) is too small, there was insufficient exploration, the algorithms
failed to identify good articles, and had a smaller number ofclicks.
On the other hand, when the parameter is too large, the algorithms
appeared to over-explore and thus wasted some of the opportunities
to increase the number of clicks.  Based on these plots on tuning
data, we chose appropriate parameters for each algorithm and ran
it once on the evaluation data in the next subsection.
Second,  it can be concluded from the plots that warm-start in-
formation is indeed helpful for finding a better match between user
interest and article content, compared to the no-feature versions of
ǫ-greedy and UCB. Specifically, bothǫ-greedy (warm)anducb
(warm)were able to beatomniscient, the highest CTRs achiev-
able by context-free policies in hindsight.  However, performance
of the two algorithms using warm-start information is not asstable
as algorithms that learn the weights online. Since the offline model
for “warm start” was trained with article CTRs estimated on all ran-
dom traffic [12],ǫ-greedy (warm)gets more stable performance
in the deployment bucket whenǫis close to1. The warm start part
also helpsucb (warm)in the learning bucket by selecting more at-
tractive articles to users from scratch, but did not helpucb (warm)
in determining the best  online for  deployment.   Sinceucbrelies
on  the a confidence  interval  for  exploration,  it is hard to correct
the initialization bias introduced by “warm start”.  In contrast, all
online-learning algorithms were able to consistently beatthe omni-
scient policy.  Therefore, we did not try the warm-start algorithms
on the evaluation data.
Third,ǫ-greedy algorithms (on the left of Figure 2) achieved sim-
ilar CTR as upper confidence bound ones (on the right of Figure2)
in the deployment bucket when appropriate parameters were used.
Thus, both types of algorithms appeared to learn comparablepoli-
cies.   However,  they  seemed  to  have  lower  CTR  in  the  learning
bucket, which is consistent with the empirical findings of context-
free algorithms [2] in real bucket tests.
Finally, to compare algorithms when data are sparse, we repeated
the same parameter tuning process for each algorithm with fewer
data, at the level of30%,20%,10%,5%, and1%.  Note that we
still used all data to evaluate an algorithm’s CTR as done in Algo-
rithm 3, but then only a fraction of available data were randomly
chosen to be used by the algorithm to improve its policy.
5.5.2    Results for Evaluation Data
With parameters optimized on the tuning data (c.f., Figure 2), we
ran the algorithms on the evaluation data and summarized theCTRs
in Table 1.   The table also reports  the CTR lift compared  to the
baseline ofǫ-greedy.  The CTR ofomniscientwas1.615, and so
a significantly larger CTR of an algorithm indicates its effective use
of user/article features for personalization. Recall thatthe reported
CTRs were normalized by the random policy’s CTR. We examine
the results more closely in the following subsections.
On the Use of Features.
We first investigate whether it helps to use features in article rec-
ommendation.   It is clear from Table 1 that,  by considering  user
features,  bothǫ-greedy (seg/disjoint/hybrid)and UCB methods
(ucb  (seg)andlinucb (disjoint/hybrid)) were  able to achieve  a
CTR lift of around10%, compared to the baselineǫ-greedy.
To better visualize the effect of features, Figure 3 shows how an
article’s CTR (when chosen by an algorithm) was lifted compared
to its base CTR (namely, the context-free CTR).
## 4
Here, an article’s
base CTR measures how interesting it is to a random user, and was
estimated from logged events.  Therefore, a high ratio of thelifted
and base CTRs of an article is a strong indicator that an algorithm
does  recommend  this article to potentially interested users.   Fig-
ure  3(a)  shows  neitherǫ-greedynorucbwas  able to  lift article
CTRs, since they made no use of user information.  In contrast, all
## 4
To  avoid  inaccurate  CTR  estimates,  only50articles  that  were
chosen most often by an algorithm were included in itsownplots.
Hence, the plots for different algorithms are not comparable.

## 0
## 1
## 2
## 3
## 0 1 2 3
lifted ctr
base ctr
## (a)ǫ-greedyanducb
## 0
## 1
## 2
## 3
## 0 1 2 3
lifted ctr
base ctr
(b) seg:ǫ-greedyanducb
## 0
## 1
## 2
## 3
## 0 1 2 3
lifted ctr
base ctr
(c) disjoint:ǫ-greedyandlinucb
## 0
## 1
## 2
## 3
## 0 1 2 3
lifted ctr
base ctr
(d) hybrid:ǫ-greedyandlinucb
Figure 3: Scatterplots of the base CTR vs. lifted CTR (in the learning bucket) of the50most frequently selected articles when100%
evaluation data were used.  Red crosses are forǫ-greedy algorithms, and blue circles are for UCB algorithms.  Note that the sets of
most frequently chosen articles varied with algorithms; see the text for details.
the other three plots show clear benefits by considering personal-
ized recommendation.  In an extreme case (Figure 3(c)), one of the
article’s CTR was lifted from1.31to3.03—a132% improvement.
Furthermore, it is consistent with our previous results on tuning
data that, compared toǫ-greedy algorithms, UCB methods achieved
higher CTRs in the deployment bucket, and the advantage was even
greater  in  the  learning  bucket.   As  mentioned  in  Section  2.2,ǫ-
greedy approaches areunguidedbecause they choose articlesuni-
formlyat random for exploration. In contrast, exploration in upper
confidence  bound  methods  are  effectivelyguidedby  confidence
intervals—a  measure  of  uncertainty  in  an  algorithm’s  CTR  esti-
mate.   Our  experimental  results imply  the effectiveness  of  upper
confidence bound methods and we believe they have similar bene-
fits in many other applications as well.
On the Size of Data.
One of the challenges in personalized web services is the scale
of the applications.  In our problem, for example, a small pool of
news articles were hand-picked by human editors.  But if we wish
to allow more choices or use automated article selection methods
to determine the article pool, the number of articles can be too large
even for the high volume of Yahoo! traffic.  Therefore, it becomes
critical for an algorithm to quickly identify a good match between
user interests and article contents when data are sparse.  Inour ex-
periments, we artificially reduced data size (to the levels of30%,
20%,10%,5%, and1%, respectively) to mimic the situation where
we have a large article pool but a fixed volume of traffic.
To better visualize the comparison results, we use bar graphs in
Figure  4  to plot  all  algorithms’  CTRs with various  data sparsity
levels.  A few observations are in order.  First, atalldata sparsity
levels, features were still useful.  At the level of1%, for instance,
we observed a10.3% improvement oflinucb (hybrid)’s CTR in the
deployment bucket (1.493) overucb’s (1.354).
Second, UCB methods consistently outperformedǫ-greedy ones
in the deployment bucket.
## 5
The advantage overǫ-greedy was even
more apparent when data size was smaller.
Third, compared toucb (seg)andlinucb (disjoint),linucb (hy-
brid)showed significant  benefits when data size was small.   Re-
call that in hybrid models, some features are shared by all articles,
making it possible for CTR information of one article to be “trans-
ferred” to others.  This advantage is particularly useful when the
article pool is large.   In contrast,  in disjoint models,  feedback  of
## 5
In the less important learning bucket, there were two exceptions
forlinucb (disjoint).
one article may not be utilized by other articles; the same istrue for
ucb (seg).  Figure 4(a) shows transfer learning is indeed helpful
when data are sparse.
## Comparingucb (seg)andlinucb (disjoint).
From Figure 4(a), it can be seen thatucb (seg)andlinucb (dis-
joint)had similar performance.  We believe it was no coincidence.
Recall that features in our disjoint model are actually normalized
membership  measures  of  a  user  in  the  five  clusters  described  in
Section  5.2.2.   Hence,  these  features  may  be  viewed  as  a  “soft”
version of the user assignment process adopted byucb (seg).
Figure  5  plots  the  histogram  of  a  user’s  relative  membership
measure to the closest cluster, namely, the largest component of the
user’s five, non-constant features.  It is clear that most users were
quite close to one of the five cluster centers:  the maximum mem-
bership of about85% users were higher than0.5, and about40% of
them were higher than0.8. Therefore, many of these features have
a highly dominating component, making the feature vector similar
to the “hard” version of user group assignment.
We believe that adding more features with diverse components,
such as those found by principal component analysis, would be nec-
essary to further distinguishlinucb (disjoint)fromucb (seg).
## 6.    CONCLUSIONS
This paper  takes  a  contextual-bandit  approach  to  personalized
web-based services such as news article recommendation. Wepro-
posed  a  simple  and  reliable  method  for  evaluating  bandit  algo-
rithms directly from logged events,  so that the often problematic
simulator-building step could  be avoided.   Based on real  Yahoo!
Front Page traffic, we found that upper confidence bound methods
generally outperform the simpler yet unguidedǫ-greedy methods.
Furthermore, our new algorithmLinUCBshows advantages when
data  are  sparse,  suggesting  its  effectiveness  to  personalized  web
services when the number of contents in the pool is large.
In the future, we plan to investigate bandit approaches to other
similar web-based serviced  such as online advertising,  andcom-
pare our algorithms to related methods such as Banditron [16].  A
second direction is to extend the bandit formulation and algorithms
in which an “arm” may refer to a complex object  rather than an
item (like an article).  An example is ranking, where an arm corre-
sponds to a permutation of retrieved webpages.  Finally, user inter-
ests change over time, and so it is interesting to consider temporal
information in bandit algorithms.

## 1
## 1.2
## 1.4
## 1.6
## 1.8
## 100%30%20%10%5%1%
ctr
data size
ε-greedy
ucb
ε-greedy (seg)
ucb (seg)
ε-greedy (disjoint)
linucb (disjoint)
ε-greedy (hybrid)
linucb (hybrid)
omniscient
(a) CTRs in the deployment bucket.
## 1
## 1.2
## 1.4
## 1.6
## 1.8
## 100%30%20%10%5%1%
ctr
data size
(b) CTRs in the learning bucket.
Figure 4: CTRs in evaluation data with varying data sizes.
## 0
## 0.05
## 0.1
## 0.15
## 0.2
## 0.25
## 0.3
## 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
maximum user membership feature
Figure 5: User maximum membership histogram.
## 7.ACKNOWLEDGMENTS
We thank Deepak Agarwal, Bee-Chung Chen, Daniel Hsu, and
Kishore  Papineni  for  many  helpful  discussions,  István  Szita and
Tom Walsh for clarifying their algorithm,  and Taylor  Xi and the
anonymous reviewers for suggestions that improved the presenta-
tion of the paper.
## 8.REFERENCES
[1]  N. Abe, A. W. Biermann, and P. M. Long. Reinforcement learning
with immediate rewards and linear hypotheses.Algorithmica,
## 37(4):263–293, 2003.
[2]  D. Agarwal, B.-C. Chen, and P. Elango. Explore/exploit schemes for
web content optimization. InProc. of the 9th International Conf. on
## Data Mining, 2009.
[3]  D. Agarwal, B.-C. Chen, P. Elango, N. Motgi, S.-T. Park,
R. Ramakrishnan, S. Roy, and J. Zachariah. Online models for
content optimization. InAdvances in Neural Information Processing
Systems 21, pages 17–24, 2009.
[4]  R. Agrawal. Sample mean based index policies witho(logn)regret
for the multi-armed bandit problem.Advances in Applied
## Probability, 27(4):1054–1078, 1995.
[5]  A. Anagnostopoulos, A. Z. Broder, E. Gabrilovich, V. Josifovski, and
L. Riedel. Just-in-time contextual advertising. InProc. of the 16th
ACM Conf. on Information and Knowledge Management, pages
## 331–340, 2007.
[6]  P. Auer. Using confidence bounds for exploitation-exploration
trade-offs.Journal of Machine Learning Research, 3:397–422, 2002.
[7]  P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the
multiarmed bandit problem.Machine Learning, 47(2–3):235–256,
## 2002.
[8]  P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. The
nonstochastic multiarmed bandit problem.SIAM Journal on
## Computing, 32(1):48–77, 2002.
[9]  D. A. Berry and B. Fristedt.Bandit Problems: Sequential Allocation
of Experiments. Monographs on Statistics and Applied Probability.
Chapman and Hall, 1985.
[10]  P. Brusilovsky, A. Kobsa, and W. Nejdl, editors.The Adaptive Web —
Methods and Strategies of Web Personalization, volume 4321 of
Lecture Notes in Computer Science. Springer Berlin / Heidelberg,
## 2007.
[11]  R. Burke. Hybrid systems for personalized recommendations. In
B. Mobasher and S. S. Anand, editors,Intelligent Techniques for Web
Personalization. Springer-Verlag, 2005.
[12]  W. Chu and S.-T. Park. Personalized recommendation on dynamic
content using predictive bilinear models. InProc. of the 18th
International Conf. on World Wide Web, pages 691–700, 2009.
[13]  W. Chu, S.-T. Park, T. Beaupre, N. Motgi, A. Phadke,
S. Chakraborty, and J. Zachariah. A case study of behavior-driven
conjoint analysis on Yahoo!: Front Page Today Module. InProc. of
the 15th ACM SIGKDD International Conf. on Knowledge Discovery
and Data Mining, pages 1097–1104, 2009.
[14]  A. Das, M. Datar, A. Garg, and S. Rajaram. Google news
personalization:  scalable online collaborative filtering. InProc. of the
## 16th International World Wide Web Conf., 2007.
[15]  J. Gittins. Bandit processes and dynamic allocation indices.Journal
of the Royal Statistical Society. Series B (Methodological),
## 41:148–177, 1979.
[16]  S. M. Kakade, S. Shalev-Shwartz, and A. Tewari. Efficient bandit
algorithms for online multiclass prediction. InProc. of the 25th
International Conf. on Machine Learning, pages 440–447, 2008.
[17]  T. L. Lai and H. Robbins. Asymptotically efficient adaptive
allocation rules.Advances in Applied Mathematics, 6(1):4–22, 1985.
[18]  J. Langford and T. Zhang. The epoch-greedy algorithm for contextual
multi-armed bandits. InAdvances in Neural Information Processing
## Systems 20, 2008.
[19]  D. J. C. MacKay.Information Theory, Inference, and Learning
## Algorithms. Cambridge University Press, 2003.
[20]  D. Mladenic. Text-learning and related intelligent agents: A survey.
IEEE Intelligent Agents, pages 44–54, 1999.
[21]  S.-T. Park, D. Pennock, O. Madani, N. Good, and D. DeCoste. Naïve
filterbots for robust cold-start recommendations. InProc. of the 12th
ACM SIGKDD International Conf. on Knowledge Discovery and
Data Mining, pages 699–705, 2006.
[22]  N. G. Pavlidis, D. K. Tasoulis, and D. J. Hand. Simulation studies of
multi-armed bandits with covariates. InProceedings on the 10th
International Conf. on Computer Modeling and Simulation, pages
## 493–498, 2008.
[23]  D. Precup, R. S. Sutton, and S. P. Singh. Eligibility traces for
off-policy policy evaluation. InProc. of the 17th Interational Conf.
on Machine Learning, pages 759–766, 2000.
[24]  H. Robbins. Some aspects of the sequential design of experiments.
Bulletin of the American Mathematical Society, 58(5):527–535,
## 1952.
[25]  J. B. Schafer, J. Konstan, and J. Riedi. Recommender systems in
e-commerce. InProc. of the 1st ACM Conf. on Electronic Commerce,
## 1999.
[26]  W. R. Thompson. On the likelihood that one unknown probability
exceeds another in view of the evidence of two samples.Biometrika,
## 25(3–4):285–294, 1933.
[27]  T. J. Walsh, I. Szita, C. Diuk, and M. L. Littman. Exploring compact
reinforcement-learning representations with linear regression. In
Proc. of the 25th Conf. on Uncertainty in Artificial Intelligence, 2009.