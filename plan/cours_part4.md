

## 1
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.LetardRL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## ESAIP –IR4 & IR5CP -IA, S8-10
## Teacher : Alexandre Letard
aletard@esaip.org
www.linkedin.com/in/alexandre-letard
## Office : D203
## Reinforcement Learning
Introduction to Reinforcement
## Learning

## 2
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Prologue
## Course Rules:
5-15’:Coursestarts5minafterappointedtime,studentshavetobereadyatthistime.
15minaftertheappointedtime,studentsarenotallowedinclassanymore.Penaltieson
yourgradeforbreakingtheruleandskippingcourses.
English written, French spoken :All course materials and practical works subjects
are written in english while speeches and explanations are given in french. Slides are
madesothatlittlenotesareneeded:takeusefulones.
Joketime,worktime:Notalksduringpresentationsexceptforinteractivetimes,low
noiselevelduringteamwork.
Definiteexit:Ifgoingoutofclassduringcourse,notcomingbackduringsession.

## 3
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Prologue
## Course Rules:
Definite exclusion:If a student get expulsed from classroom by the teacher at any
pointoftime,noneedtocomebackinfuturesessions.
Partitioning:Courses, meeting in break times, projects tutoring, disciplinary
committeeandevaluationarealldifferentmatters.Studentsattitude/relationshipwith
teacher will not influence the evaluation of their work, likewise their grades will not
definetheopinionoftheteacheraboutthestudentsandreciprocally.
Honesty and Respect:Anything can be talked about as long as it is done in a
respectful way, be it about the course contents, its process or any other matter the
studentsorteacherwouldliketotalkabout.

## 4
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Prologue
Learning outcomes
I-Fundamentallevel:Generalknowledgeoverreinforcementlearning:general
framework, main techniques including their principles, strengths and
weaknesses.
III–«NearExpert»level:Implementingreinforcementlearningalgorithms
fromscratchusingonlynumpyandpandaslibrariesfollowingresearchpapers.
Tothisend,theuseofgenerativeAIisstrictlyprohibitedforthiswhole
course,includinghomework.

## 5
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Prologue
Course organisation
## 16sessions(24h),splitinto3months
## Oursessionsdates,boldforfirst,lastandevaluationssessions:
Coefficient1(outof4)forUES8-4Majeure2
## •
## 22/01/2026(2)
## •
## 23/01/2026(3)
## •
## 16/02/2026(2)
## •
## 20/02/2026(3)
## •
## 10/03/2026(2)
## •
## 30/03/2026(2)
## •
## 03/04/2026(2)

## 6
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Prologue
## Evaluation
Learning Activities:Listening, Course Participation, Practical work realization,
HomeworkandAssiduity,worth20%ofyourfinalgrade
TheoreticalEvaluations:moodleQCMsonsomerandomsessions,worth30%ofyourfinal
grade
Retake exams: Additionnal time on projects according to provided feedback with higher
requirements.
## 
## Projectswillbepresentedforevaluationonlasttwosessions:03/04/2026
Mini-Projects:Specificswillbegivenatalatertime,worth50%ofyourfinalgrade

## 7
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
1.Prologue
- Introduction to ReinforcementLearning
- Multi-ArmedBandits (MABs)
- ContextualMulti-ArmedBandits (CMABs)
- MABs/CMABsExtensions
## 6. Projects
## 7. Course Conclusion
## Prologue
## Syllabus
N.B: This course aims at exploring the path towards expertise over an AI technique, though much
moretimewouldbeneededtoactuallyreachit.

## 8
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Prologue
Course organisation
## •
## 22/01/2026(2)
## •
## 23/01/2026(3)
## •
## 16/02/2026(2)
## •
## 20/02/2026(3)
## •
## 10/03/2026(2)
## •
## 13/03/2026(2)
## •
## 03/04/2026(2)
## •
## 27/03/2026(0)
Prologue, RL introduction, RL techniques Exploration
Multi-ArmedBandits (MAB), practicalwork
RL techniques –Reversedclass
ContextualMulti-ArmedBandits (CMAB), practicalwork
MABs & CMABS extensions, Mini-Projects
Mini-Projects
Mini-projectsdefenseand course conclusion
Submissiondeadline for projects(to check togetheron last session)
N.B:Thepracticalworksconductedduringcoursewillbethebasisforyourprojects.

## 9
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Prologue
## Questions ?

## 10
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## 1. Prologue
2.Introduction to ReinforcementLearning
- Multi-ArmedBandits (MABs)
- ContextualMulti-ArmedBandits (CMABs)
- MABs/CMABsExtensions
## 6. Projects
## 7. Course Conclusion
## Prologue
## Syllabus

## 11
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Reinforcement Learning: Motivations
Imagine you want to design a self-driving boat, how
wouldyoudo?
## ➢
Lack of data for boats, deep learning and data-
drivenMLcannotbeapplied.
## ➢
Navigation behaviors are hard to modelized on
rules-based systems (not set in stone), rules-
basedsystemscannotbeapplied.
Introduction to Reinforcement Learning
Reinforcement Learning relies on online-learning,
applicablewhenpreviousapproachescannot.
Do not try to design a self-driving boat using RL from
scratchthough...

## 12
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Reinforcement Learning: Applications
Introduction to Reinforcement Learning
## ➢
Games:Training performed by letting 2 agents play against each other
longandoftenenough.Initialdatacanbefoundtofostertheirtraining.
o
Examples:Chess,Go,BoardGames,NPCwithadaptivebehaviors...
## ➢
Robotics:Mostly deep reinforcement learning. For most existing
applications,datacanbefoundorsimulatorscanbebuilt.
o
Examples:Robot control (grasping and manipulating objects,
navigation),autonomousvehicles...
## ➢
Recommender systems:Everywhere nowadays. RL can be used for
personalizedrecommendationsorinterfacesoptimization.
o
Examples:LinkedIn,Netflix,Leclercdrive,Tinder,Amazon...
## ➢
## ....

## 13
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Reinforcement Learning: When and why ?
Introduction to Reinforcement Learning
## ➢
## Tolaunchaninnovativeapplicationforwhichhistoricaldataishardtofind.
## ➢
## Whenaimingforonlineimprovement.
## ➢
Fordynamic/Non-stationnaryenvironments.
## ➢
Scenarios involving sparse rewards/feedback.
## ➢
Exploration-Exploitation tradeoff.

## 14
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
AI Techniques: Machine Learning
SupervisedLearning    UnsupervisedLearning
## Data:(x,y)
xisdata,yislabel
## Data:x
xisdata,nolabels
## Goal:learningmap
x →y
Goal:Undercoverdata
structure
## Example:
## Example:
## Thisisabook.
Thisthing issimilarto thatthing.
Semi-supervisedLearning
## +
ReinforcementLearning
## Data:state-actionpairs(s,a)
Goal:Maximizing long-term
cumulatedrewards
## Example:
Read this to know better and
becauseAIisfun.
AI Concepts Review & Evaluation Methods

## 15
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Reinforcement Learning: A brief history
Introduction to Reinforcement Learning
## ➢
1950-1960: groundwork for RL with dynamic programming and optimal control theory
(Bellman).
## ➢
1970-1980:FirstRLalgorithms,Q-Learning(Watkins),TemporalDifference(Sutton).
## ➢
1990-2000:FirstworksinvolvingdeepRL,NeuralFittedQ-iteration(NFQ)algorithm.
## ➢
2010-Present: Numerous breakthroughs with Deep Q-Networks (DQN) and their
various applications.

## 16
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Reinforcement Learning: An illustrative example
Introduction to Reinforcement Learning
## ➢
This dog is learning. Also, he wants to be petted, to eat
lotoffoodandnottobepunished.
## ➢
## Intheenvironment,thereisagirlandfood.
## Whatshouldthedogdo?

## 17
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Reinforcement Learning: Key Concepts & General Framework
Introduction to Reinforcement Learning
## ➢
## Agent:orlearningagent,theentityresponsiblefortakingactionsinthe
environmentimprovingitsdecision-makingovertime(thedog).
## ➢
Environment: the system with which the agent interacts (the world
around the dog), which can be on different states, leading to different
rewardsorlossfortheagent’sactions.
## ➢
## State(s):currentconfigurationoftheenvironment(agirlwithfood,no
food), capturing all relevant information for decision-making at a given
timestep.
## ➢
Action(s): the decision made by the agent (eating the food, waiting),
leadingtoanewstate(foodinthedog’smouth,punishment).

## 18
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Reinforcement Learning: Key Concepts & General Framework
Introduction to Reinforcement Learning
## ➢
Policy(π):Amappingfromstatestoaction(thedogthinkingabouteating
thefoodorwaiting),specifyingthestrategyappliedbytheagent.
## ➢
Reward (r): Feedback provided by the environment for the agent’s action
## (foodeaten,givenorpunishment),translatethedesirabilityoftheaction.
## ➢
Valuefunction(V):predictstheexpectedrewardanagentcanachievefrom
agivenstateunderaparticularpolicy(howmuchpettingandfoodthedog
can expect by waiting when the girl has food or not). Quantifies the
desirabilityofstatesorstate-actionpairs.
## ➢
## Model:representstheagent’sknowledgeoftheenvironment’sdynamics(the
girlusuallydoesnotpunish).Canbeusedtopredictsrewardsandnextstate
oftheenvironment.

## 19
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Reinforcement Learning: Key Concepts & General Framework
Introduction to Reinforcement Learning
AgentEnvir.
## State
## Action
## Reward
RLproblemsareoftendescribedasMarkovDecisionProcesses:
## ➢
Initialization:Initializetheagent'spolicy,valuefunction,and
otherparameters.
## ➢
Interaction:The agent selects actions based on its current
policy and observes the resulting states and rewards from the
environment.
## ➢
Learning:Updatetheagent'spolicyandvaluefunctionbased
onobservedexperiencestoimprovedecision-making.
## ➢
Evaluation:Assess the performance of the learned policy
throughsimulationorreal-worldinteraction.
## 푠
## ௧
## 푎
## ௧
## 푟
## ௧
Repeated process for all
iterations t of a problem
instance.

## 20
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Reinforcement Learning: Main Methods
Introduction to Reinforcement Learning
## 1.
Action-ValueMethods
Principle:Value-basedmethodsfocusonfiguringoutwhichactionsarethebesttotakeineachsituationby
estimatingthe"value"ofbeingindifferentstatesortakingdifferentactions.
Keyalgorithms:Q-learning,DeepQ-Networks(DQN),DoubleQ-learning
Strengths:Weaknesses:
## •
## Handlelargestateandactionspaces;
## •
## Efficientforproblemswithdiscretesetofactions.
## •
## Sensitivetohyperparameters,extensivetuning;
## •
## Strugglewithcontinuous,largestatespacesand
stochasticpolicies.
Example:Thinkofitliketryingtodecidewhichroutetotaketoschooleachmorningbasedonpastexperiencesof
howlongeachrouteusuallytakes.
Applications:Boardgames,robotcontrolandnavigation,decisionmakingindiscreteactionspace.
Requirements:Well-definedstatesandactions,abilitytoaccuratelyestimatevaluefunctions.

## 21
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Reinforcement Learning: Main Methods
Introduction to Reinforcement Learning
## 2.
PolicyGradientMethods
Principle:Policygradientmethodsusegradientascenttoupdatetheparametersofthepolicynetworkinthe
directionthatincreasesexpectedcumulativereward.
Keyalgorithms:REINFORCE(MonteCarloPolicyGradient),ProximalPolicyOptimization(PPO)
Strengths:Weaknesses:
## •
## Handlestochasticpoliciesandcontinuousactionspaces;
## •
## Oftenconvergefasterthanvalue-basedmethods.
## •
## Sufferfromhighvarianceingradientestimates;
## •
## Requirecarefultuningoflearningrates.
Example:It's like learning to play a video game by trying different moves and seeing which ones result in the
highestscores,thengettingbetteratchoosingthosemovesovertime.
Applications:Roboticmanipulationandcontrol,continuouscontroltaskssuchasautonomousdriving.
Requirements:Accesstoasimulatororreal-worldenvironmentfortraining.

## 22
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Reinforcement Learning: Main Methods
Introduction to Reinforcement Learning
## 3.
Actor-CriticMethods
Principle:Thecriticlearnsvaluefunctionstoevaluateactions,whiletheactorlearnsapolicythatmaximizes
expectedcumulativerewardbasedonthesevalueestimates
Keyalgorithms:DeepDeterministicPolicyGradient(DDPG),AdvantageActor-Critic(A2C),A3C
Strengths:Weaknesses:
## •
## Handlebothdiscreteandcontinuousactionspaces;
## •
## Morestablethanpurepolicygradientmethods.
## •
Complex architecture and hyperparameter
tuningcomparedtovalue-basedmethods.
Example:Imaginehavingafriend(thecritic)whogivesyouadviceonwhichmovestomakeinagame(theactor),
helpingyouimproveyourperformance.
Applications:Roboticandautonomoussystems,gameplayingandstrategyoptimization.
Requirements:Accesstoasimulatororreal-worldenvironment.

## 23
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Reinforcement Learning: Main Methods
Introduction to Reinforcement Learning
## 4.
Model-BasedMethods
Principle:Model-based methods learn a predictive model of the environment's dynamics from observed
experiencesanduseitforplanningordecision-making.
Keyalgorithms:ModelPredictiveControl(MPC),Dyna-Q,MonteCarloTreeSearch(MCTS)
Strengths:Weaknesses:
## •
## Improveefficiencybysimulatingfuturetrajectories;
## •
## Handlesparseordelayedrewardsmoreeffectively.
## •
## Relyonaccuratemodelsoftheenvironment;
## •
## Canbecomputationallyexpensive.
Example:It'slikelearningtherulesofanewboardgamesoyoucanthinkaheadandpredicttheconsequencesof
differentmoves.
Applications:Roboticandautonomoussystems,gameplayingandstrategyoptimization,adaptiveuserinterfaces.
Requirements:Access to an environment for training and validation and ability to learn accurately
environmentdynamics.

## 24
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Reinforcement Learning: Main Methods
Introduction to Reinforcement Learning
## 5.
TemporalDifferenceLearningMethods
Principle:TD learning updates value estimates based on observed transitions and immediate rewards, using
thedifferencebetweencurrentandpredictedvalues(temporaldifference)astheupdatesignal.
Keyalgorithms:SARSA,TD(λ),Q-learning
Strengths:Weaknesses:
## •
## Learnonline,hencesuitableforreal-timeapplications;
## •
## Handlepartiallyobservableenvironments.
## •
## Tuningoflearningratesandexplorationstrategies;
## •
## Cansufferfromhighvarianceinvalueestimates.
Example:It'slikeguessinghowlongitwilltakeyoutowalktoafriend'shouse,thenupdatingyourguessasyougo
basedonhowfastyou'reactuallywalking.
Applications:Dialoguegeneration,gameplayingandoptimization,roboticsandautonomoussystems.
Requirements:Accesstoan environmentfortrainingand validationandability toestimate value functions
accurately.

## 25
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Reinforcement Learning challenges
Introduction to Reinforcement Learning
## ➢
Explorationvs.Exploitation:Findingtherightbalancebetweenexploringnew(suboptimal)actions
andexploitingknownactionstomaximizerewards.
## ➢
SampleEfficiency:Learningefficientlyfromlimiteddataisasignificantchallenge,especiallyinhigh-
dimensionalstateandactionspacesastrial-and-errorlearningcanbedata-intensiveandslow.
## ➢
Generalization:Generalizinglearnedpoliciesorvaluefunctionstonew,unseensituationsthoughthe
agentexperiencesareoftencontextspecific.
## ➢
SafetyandEthicalConcerns:EnsuringthesafetyandethicaluseofRLsystems,particularlyincritical
domainslikehealthcareandautonomousvehiclesasRLcanlearnundesirablebehaviorsasshortcutsfor
rewards.

## 26
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Summary
Introduction to Reinforcement Learning
## ➢
Reinforcementlearningdiffersfromotherapproachesbyitsemphasisontheagentlearningfromdirect
interaction with the environment, in an iterative process of trials and errors, without relying on
supervisedlearning.
## ➢
RLusesaformalframeworkdefiningtheinteractionbetweenthelearningagentanditsenvironmentin
termsofstates,actionsandrewards.
## ➢
Themodel representingtheknowledgeoftheagentonitsenvironmentandthevaluefunctionusedto
predict an expected cumulative reward for state-action pairs help the agent adjusting its policy, the
strategyfollowedtochooseanactionatanytimestep.
## ➢
While there is an extensive literature over RL with each variant involving different requirements and
purposes, those core principles are shared by all of them or most.

## 27
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Questions ?
Introduction to Reinforcement Learning

## 28
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Reversed Class:
RL Techniques Exploration
Students work by teams of3 students on a sub-field of reinforcementlearning (e.g. model-basedmethods), each student have to
study 1 algorithm of the subfield (e.g. SARSA). Each team prepare a presentation to explain the workings of these algorithms,
startingbythesharedconcepts(applicabletoall3algorithms)andthenthespecificsofeach.
The used sources, including the original paper of a presented algorithm must be cited in the presentation. As much as possible
studentsshouldunderstandtheoriginalpaper.
Presentationswillbeholdthe 16/02/2026,eachshould be timedbetween 15to20min.Allpresentationsmustbesubmittedon
moodleonthe13/02/2613hatthelatesttobesharedwitheveryoneafterthereversedclass. Teamsmustberegisteredbeforehand
withselectedalgorithmsusingthetablesharedonmoodle.Asamealgorithmcanbestudiedbyatmost2teams.
Basedonthepresentationquality,studentswillearnupto2pointsbonusfortheoreticalevaluation.Latertheoreticalexamsmay
involvequestionsrelatedtothesepresentations.
## Data Preparation

## 29
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## 1. Prologue
- Introduction to ReinforcementLearning
3.Multi-ArmedBandits (MABs)
- ContextualMulti-ArmedBandits (CMABs)
- MABs/CMABsExtensions
## 6. Projects
## 7. Course Conclusion
Introduction to Reinforcement Learning
## Syllabus

## 30
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Why Mabs - Applications
Multi-Armed Bandits (MABs)
“There are many reasons to care about bandit problems. Decision-
making with uncertainty is a challenge we all face, and bandits
provide a simple model of this dilemma.”
TorLattimoreandCsabaSzepesvàri,2020

## 31
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Why Mabs - Applications
Multi-Armed Bandits (MABs)
## ➢
## Healthcare:clinicaltrials
## ➢
Finance: portfolio selection with risk
awareness
## ➢
Dynamic pricing: defining real-time prices
foronlineretainers
## ➢
## Recommender Systems: Exploration-
exploitationtradeoffforusers’preferences
## ➢
InfluenceMaximization:Socialnetworks
[Bou20] Djallel Bouneffouf, Irina Rish, and Charu Aggarwal. 2020. Survey on Applications of Multi-Armed and Contextual
Bandits.In2020IEEECongressonEvolutionaryComputation(CEC).IEEEPress,1–8.
## Non-
stationnary
## CMAB
CMABNon-
stationnary
## MAB
## MAB
XXHealthcare
XFinance
XDynamic Pricing
XXXXRecommander
## Systems
XInfluence
## Maximization
XAnomaly
detection

## 32
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
MABs - Principle
Multi-Armed Bandits (MABs)
## Imageextractedfrom:http://www.primarydigit.com/blog/archives/12-2015
## ➢
Analogy:“Agambleratarowofslotmachineshastodecide
which machines to play, how many times to play each
machineandinwhichordertoplaythem.Whenplayed,each
machine provides a reward from a distribution specific to
that machine. The objective is to maximize the sum of
rewardsearnedthroughasequenceofleverpulls.”
## ➢
Exploitation-Explorationexample: To maximize your use
of the streaming platform, should we recommend you
contentfromyourfavoritestreameroranewone,andwhen?

## 33
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Stochastic MABs – Single State Markovian process
Multi-Armed Bandits (MABs)
AgentEnvir.
## State
## Action
## Reward
## 푠
## ௧
## 푎
## ௧
## 푟
## ௧
## ➢
## Formaldefinition:
## •
## Singlestate
## 푆 ={푠}
## •
## 풜 =푎
## ଵ
## ,...,푎
## ௠
## :setofactions,referredas“arms”
## •
## Spaceofrewards(oftenin[0,1])
## ➢
## Notransitionfunctiontolearn,onlythestochasticrewardfunction
## ➢
## Sequentialdecision-makingproblemwhere,ateachiterationttheagent:
## •
## Observestheenvironmentstate
## •
## Performsanaction
## 풂
## 풕
## •
## Observestheassociatedrewardandupdateitspolicyπ

## 34
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
MABs - Environment
Multi-Armed Bandits (MABs)
## 풜 ={풂
## ퟏ
## ,...,풂
## 풎
## }
: available ground arms
## 퐃= 흁
## ퟏ
## ,...,흁
## 풎
## ∈[ퟎ,ퟏ]
## 풎
: rewards expectations
## 흁
## 풊,풕
## =피[풓
## 풕,풂
## ]
: average observed reward,
## 흁
## 풊,풕
## =
## ∑
## 풓
## 풂
## 풊
## ,풕
## ᇲ
## 풕
## 풕
## ᇲ
## సퟎ
## 풕
## 풂
## 풊
## 
An optimal policy π* knowsexact rewardsdistribution
## 흁
## ∗
and can thusperformsoptimal
action a* at eachround.
Where each arm
## 푎
## ௜

corresponds to a specific actionwhich can be performed by the agent.

## 35
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
MABs – Objective and Metrics
Multi-Armed Bandits (MABs)
## 
## Minimizing Cumulated Regrets:
## 휌푇 =푇휇
## ∗
## −෍푟
## ௧
## ்
## ௧ୀଵ
## 
Or equivalently, maximizing Global Accuracy:
## Acc푇=
## ∑
## ௥
## ೟
## ೅
## ೟సభ
## ்
## Where:
T : Horizon (numberof iterationsperformed)
## 휌푇
: Cumulatedregret afterhorizon
## 휇
## ∗
## =max
## ௔
## 휇
## ௔
reward expectation for optimal arm
## 푟
## ௧
## :
Observed reward for action performed at round t

## 36
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
UCB1 Algorithm
Multi-Armed Bandits (MABs)
Ateachround,UCBselectaction
## 풂
## 풕
suchthat:
## 풙
## 풋
## ഥ
## +
## ퟐ퐥퐧풏
## 풏
## 풋
## 퐚
## 퐭
## =퐚퐫퐠퐦퐚퐱 풙
## 풋
## ഥ
## +
## ퟐ풍풏풏
## 풏
## 풋

UpperConfidence Bound
## (nusuallynotedt,
numberof iterations)
Note:Theupperconfidenceboundiscomputedas
a fraction of alogarithmic term with a linear term,
translating the idea that uncertainty is decreasing
over time. This implies thatexploration is
decreasingovertimeaswell.
## 풍풊풎
## 풏→ஶ
## ퟐ풍풏풏
## 풏
## 풋
## =
## =ퟎ
Average observedrewardfor
arm j (usuallynoted
## 흁
## 풕,풋
## ෞ
## )

## 37
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
UCB1 Algorithm – Illustrative example
Multi-Armed Bandits (MABs)
## Meanobservedrewardand
estimatedupperboundfor 4 arms
aftera few  iterations
## Meanobservedrewardand
estimatedupperboundfor 4 arms
aftersomemore iterations
Imageextractedfrom:https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/

## 38
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
UCB1 Algorithm
Multi-Armed Bandits (MABs)
## ➢
UpperConfidenceBound:Thenamecomesfromtheideaofcalculatinganupperconfidencebound
foreach action'sestimated reward.Thisupperconfidence bound representsouruncertaintyaboutthe
truerewardoftheaction.
## ➢
Action Selection:At each time step, the UCB algorithm selects the action with the highest upper
confidence bound. This selection process, said to be optimistic in regards to uncertainty, is how the
algorithmhandletheexploitation-explorationtradeoff.
## ➢
ExplorationParameter:Thealgorithmincludesanexplorationparameterλ(usuallysetsuchthatλ=2)
that determines the balance between exploration and exploitation. A higher exploration parameter
encouragesmoreexploration,whilealoweronefavorsexploitation.
## ➢
Experienced Note: UCB-based algorithms tend to be very sensitive to the program early stage (first
rounds).Theperformancesafteragoodorbadstartcanbesignificantlydifferent,henceseveralrunsare
recommendedtoensureaproperobservation.

## 39
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Thompson-Sampling Algorithm
Multi-Armed Bandits (MABs)
## 
On the first definition of the algorithms, theoretical proofs were conducted for
## 푟∈{0,1}
(Bernoulli rewards), hence
## 푺
## 풊
depicted the number of times the action
## 풂
## 풊
was rewarded
## (successes)and
## 푭
## 풊
depictedthetotalnumberoflossesoccurredbyplayingit.
## 
Sincethen,TSalgorithmhasbeenprovenwithsub-linearregretforrewards
## 푟∈0,1
## ,allowinga
more general setting where
## 푺
## 풊
can be seen as the total amount of rewards observed for
playing
## 풂
## 풊
and
## 푭
## 풊
itscumulatedregrets.

## 40
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Thompson-Sampling Algorithm – Sampling illustration
Multi-Armed Bandits (MABs)
Imageextractedfrom:https://towardsdatascience.com/thompson-sampling-fc28817eacb8
## 
For each arm, one beta
distribution of rewards estimated
basedon
## 푺
## 풊
and
## 푭
## 풊
## .
## 
At each round,TS randomly
(and uniformly) sample a
reward expectation for each
arm(dots)and play the arm
withhighestvalue(bluehere).
Y-axis:Densityofsamples;X-axis:rewardprobability-value
## Betadistributionwithmeanvalueof:
## ஑
## ஑ା ஒ
## =
## 풏풃 풔풖풄풄풆풔풔풆풔
## 풏풃 풕풓풊풂풍풔


## 41
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Thompson-Sampling Algorithm – Sampling illustration
Multi-Armed Bandits (MABs)
Imageextractedfrom:https://towardsdatascience.com/thompson-sampling-fc28817eacb8
## 
Based on the observed reward
after performing action, the
posterior distribution of the
playedarmisupdated.
Y-axis:Densityofsamples;X-axis:rewardprobability-value
## Betadistributionwithmeanvalueof:
## ஑
## ஑ା ஒ
## =
## 풏풃 풔풖풄풄풆풔풔풆풔
## 풏풃 풕풓풊풂풍풔


## 42
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Thompson-Sampling Algorithm – Sampling illustration
Multi-Armed Bandits (MABs)
Imageextractedfrom:https://towardsdatascience.com/thompson-sampling-fc28817eacb8
## 
## Notethattheuniformsampling
allow the exploration(e.g. 4th
figure,greenisplayed).
## 
Hence, theleast observations
performed,thewidertherange
of the distributions-> the
higher  the  exploration
probability.
Y-axis:Densityofsamples;X-axis:rewardprobability-value
## Betadistributionwithmeanvalueof:
## ஑
## ஑ା ஒ
## =
## 풏풃 풔풖풄풄풆풔풔풆풔
## 풏풃 풕풓풊풂풍풔


## 43
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Thompson-Sampling Algorithm – Sampling illustration
Multi-Armed Bandits (MABs)
Imageextractedfrom:https://towardsdatascience.com/thompson-sampling-fc28817eacb8
## 
As the algorithm performs more
trials, theuncertainty about the
reward distribution decreases,
resulting innarrower ranges for
the distributions->lower
exploration probability (higher
exploitation).
Y-axis:Densityofsamples;X-axis:rewardprobability-value
## Betadistributionwithmeanvalueof:
## ஑
## ஑ା ஒ
## =
## 풏풃 풔풖풄풄풆풔풔풆풔
## 풏풃 풕풓풊풂풍풔


## 44
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Thompson-Sampling Algorithm
Multi-Armed Bandits (MABs)
## ➢
Bayesian approach:Instead of directly estimating the rewards of different actions like in the UCB
algorithm,itmaintainsaprobabilitydistributionoverthepossiblerewardoutcomesforeachaction.
## ➢
Sampling from Posterior Distribution:At each time step, Thompson Sampling samples a reward
distribution for each action from its posterior distribution based on the observed data. The posterior
distributionisupdatedusingBayes'theorem,incorporatingnewobservations
## ➢
Action Selection:Once the reward distributions are sampled, Thompson Sampling selects the action
withthehighestsampledreward.
## ➢
Exploration: Similar to UCB, Thompson Sampling naturally balances exploration and exploitation.
Actions with uncertain rewards have wider posterior distributions, encouraging exploration, while
actionswithhigherexpectedrewardshavenarrowerdistributions,promotingexploitation.
## ➢
Experiencednote:Thoughtheoldestone,TSisusuallymoreefficientandrobustthanotherstochastic
MABs.

## 45
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Summary
Multi-Armed Bandits (MABs)
## ➢
MABsalgorithmsarereinforcementlearningapproachesdesignedasMarkovianprocesseswithasingle
state,aimingat strikingabalancebetweenexplorationandexploitation.Theyhavebeenappliedina
wide range of applications thanks to their ease of use, low requirements and strong theoretical
guarantees.
## ➢
Inthiscourse,wecoveredthestochasticmulti-armedbanditproblem,wheretherewardsareconsidered
i.i.dvariables.Thearmselectionandexplorationprocessesarespecifictoeachalgorithm,butmostlyrely
inacountingofthenumberofobservedrewardsoverthenumberoftrials.
## ➢
The most important thing to remember on this part is the process of understanding
algorithms from their original paper. Ask yourself if you understood the underlying of the
algorithmwell,readthepartofthepaperintroducingitagainandcheckhowyoucouldhave
understoodbetter.

## 46
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Questions ?
Multi-Armed Bandits (MABs)

## 47
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Practical Work :
UCB1, Thompson Sampling
Download archive file for MABs simulator on moodle. Explore the software and, using
theprovidedpapers,implement(usingpandasandnumpylibraries):
## -UCB1
-ThompsonSamplingForreference,youcanrefertoε-greedyperformances.
Multi-Armed Bandits (MABs)

## 48
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## 1. Prologue
- Introduction to ReinforcementLearning
- Multi-ArmedBandits (MABs)
4.ContextualMulti-ArmedBandits (CMABs)
- MABs/CMABsExtensions
## 6. Projects
## 7. Course Conclusion
Introduction to Reinforcement Learning
## Syllabus

## 49
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Practical Work :
LinUCB, CTS
## Usingtheprovidedpapers,implement(usingpandasandnumpylibraries):
-LinUCB(p4,algorithm1,disjointlinearmodel)
-AfterLinUCBvalidation,LinTS/CTS(p3,algorithm1)
-AfterCTSvalidation:generalcontextualframeworkandcontextualgreedy
## Expectedperformancesforreference:
LinUCB:RSASM:0.78;PokerHand:0.53,Covertype:0.72,Mushrooms:0.998
CTS:RSASM:0.70;PokerHand:0.52,Covertype:0.72,Mushrooms:0.996
Contextual Multi-Armed Bandits (CMABs)

## 50
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
CMABs - Concept
Contextual Multi-Armed Bandits (CMABs)
[LZ08] J. Langford et T. Zhang, “The epoch-greedy algorithm for multi-armed bandits with side information”, NIPS, 2008
## 피풓
## 풕,풂
## 풙
## 풕
## ]
## =
## 휽
## ෡
## 풕,풂
## ୃ
## 풙
## 풕
## [LZ08]
In thecontextual case, it is
considered  that  thereward
expectationofarm
## 푎 ∈풜
islinearly
dependentofobservedcontext
## 푥
## ௧
## :
## Where
## 휽
## ෡
## 풕,풂
## ୃ
## ∈ ℝ
## 풅
is acoefficient vector
associatedtoarma,initializedasanull-vector
andestimatedateachround/iteration.

## 51
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
CMABs - Concept
Contextual Multi-Armed Bandits (CMABs)
[LZ08] J. Langford et T. Zhang, “The epoch-greedy algorithm for multi-armed bandits with side information”, NIPS, 2008
## 0110...010
## 00.6750.780.002...0.0450.340.011
## 푎
## ௧
## = 푎푟푔푚푎푥  피풓
## 풕,풂
## 풙
## 풕
## ]
## 풙
## 풕
## 휽
## ෡
## 풕,풂
## Walkin Broceliandeforest
## 피풓
## 풕,풂
## 풙
## 풕
## ]
## =
## 휽
## ෡
## 풕,풂
## ୃ
## 풙
## 풕
## [LZ08]
In thecontextual case, it is
considered  that  thereward
expectationofarm
## 푎 ∈풜
islinearly
dependentofobservedcontext
## 푥
## ௧
## :
## Where
## 휽
## ෡
## 풕,풂
## ୃ
## ∈ ℝ
## 풅
is acoefficient vector
associatedtoarma,initializedasanull-vector
andestimatedateachround/iteration.

## 52
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Linear ε-Greedy algorithm
Contextual Multi-Armed Bandits (CMABs)
At eachround thereare:
## 
A probability
## 휺∈[ퟎ,ퟏ]
to randomlyexplore solution space:
## 
A probability
## ퟏ−휺
to exploit by selectingthe arm with
highestrewardexpectation knowing
## 풙
## 풕
## :
## 풂
## 풕
## = 푎푟푔푚푎푥  피풓
## 풕,풂
## 풙
## 풕
## ]=푎푟푔푚푎푥 (휃
## መ
## ௧,௔
## ୃ

## 푥
## ௧
## )
## 풂
## 풕
## =푅푎푛푑표푚(풜)
## Exampleforprevioususer:recommendingakayakexpedition
Exampleforprevioususer:recommendingawalkinBroceliandeforest

## 53
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Linear ε-Greedy algorithm – Theta computation
Contextual Multi-Armed Bandits (CMABs)
Note that
## 푓
## ௔,௧
isa convenientvariable updatedusingthe
element-wise(Hadamar) productof the observedreward
## 푟
## ௧
and the contextvector
## 푥
## ௧
## .
## Wecouldwrite:
## =
## 퐵
## ௔,௧
## ିଵ
## 푓
## ௔,௧
## 휃
## ෠
## ௔,௧
## =
## 퐵
## ௔,௧
## ିଵ
## ෍푟
## ௧
## 푥
## ௧
## ்
## ௧ୀଵ

## 54
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Linear ε-Greedy algorithm – Covariance matrix
Contextual Multi-Armed Bandits (CMABs)
CovarianceMatrix(B):Inthecontextualcase,thismatrixcapturestherelationshipsbetweenthe
featuresinthecontextvector
## 푥
## ௧
andisupdatedaftereachiterationusingtheouterproductof
thecontextvector:
## 퐵
## ௔,௧
## = 퐵
## ௔,௧
## +푥
## ௧
## 푥
## ௧
## ்
## Notethatthecovariancematrixisinitializedastheidentitymatrixandthat
## 휃
## መ
## ௧,௔
## ୃ
iscomputedusing
theinversematrix
## 퐵
## ௔,௧
## ିଵ
formorestablecomputationandpreventingoverfitting.

## 55
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Linear UCB (LinUCB) Algorithm
Contextual Multi-Armed Bandits (CMABs)
## 푝
## ௔,௧
## =휽
## ෡
## 풂
## 푻
## 풙
## 풂,풕
## +휶풙
## 풂,풕
## 푻
## 푨
## 풂
## ିퟏ
## 풙
## 풂,풕

UpperConfidence Bound in
regards to providedcontext
Expectedrewardfor arm a
givenprovidedcontext
## ➢
## Where
## 휶
isatuningparameterforexploration-
exploitationtradeoff.Usually:
## 훼 =1+
ln2 / 훿
## 2
## With
## 휹∈ퟎ,ퟏ (=ퟎ,ퟎퟏ)

## 56
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Linear UCB (LinUCB) Algorithm
Contextual Multi-Armed Bandits (CMABs)
## ➢
LinearModel:eachaction(orarm)isassociatedwithasetoffeatures.Thealgorithmassumesthatthereexistsa
linearrelationshipbetweenthesefeaturesandtheexpectedrewardforeachaction.
## ➢
Upper Confidence Bound:Instead of directly estimating uncertainty about rewards, LinUCB estimates
uncertaintyabouttheparameters
## 휃
## ௜
ofthelinearmodel.
## ➢
ActionSelection:Ateachtimestep,LinUCBselectstheactionwiththehighestupperconfidencebound.The
upper confidence bound is calculated based on the estimated parameter vector
## 휽
## ෡
## 풂,풕
and its associated
uncertainty.
## ➢
Exploration Parameter: The algorithm includes an exploration parameter
## 훼
that determines the balance
between exploration and exploitation. A higher exploration parameter encourages more exploration, while a
loweronefavorsexploitation.
## ➢
ExperiencedNote:UCB-based algorithmstend to be very sensitive tothe programearlystage(firstrounds),
thisextendstothecontextualcaseaswell.

## 57
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Contextual Thompson-Sampling (CTS) Algorithm
Contextual Multi-Armed Bandits (CMABs)
## 푝
## ௔,௧
## =휇̅푡 =푁(휇ො,푣
## ଶ
## 퐵
## ିଵ
## )

## Expectedrewardsgiven
providedcontext
## Inverse
covariance matrix
Normal distribution,
oftenusedfor
continuousrewards
Variance Parameterv: highervalue for more exploration, typicallyset
suchthat:
## 휈 =휎
## 24
## 휖
## 푑ln
## 1
## 훿
## With:
## 휖 =
## 1
ln (푇)
## ,
## 훿 ∈[0,1]

## 58
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Contextual Thompson Sampling (CTS) Algorithm
Contextual Multi-Armed Bandits (CMABs)
## ➢
LinearModel:eachaction(orarm)isassociatedwithasetoffeatures.Thealgorithmassumesthatthereexistsa
linearrelationshipbetweenthesefeaturesandtheexpectedrewardforeachaction.
## ➢
Sampling from posterior:In CTS, instead of estimating uncertainty about the parameters directly as in
LinUCB,thealgorithmsamplesfromtheposteriordistributionovertheparameters.Thisposteriordistribution
capturestheuncertaintyabouttheparametersgiventheobserveddataandthepriordistribution.
## ➢
ActionSelection:LikeoriginalTS,selectstheactionwithhighestexpectedrewardsfromthesampledvalues.
## ➢
ExplorationParameter:Thealgorithmincludesanexplorationparameter
## 휈
.Highervalueswillleadtohigher
varianceinthesampledparametervectors,encouragingmoreexploration.
## ➢
ExperiencedNote:WhileLinTS/CTSfrequentlyofferniceperformances,thealgorithmdonotscaleupwell
withthenumberofarmsduetoahighcomputationalcost.

## 59
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
Whatisthe weightof each
featuresfor reward?
CMAB Algorithms
Contextual Multi-Armed Bandits (CMABs)
## ➢
## Notationcleanupandmainroles:
## 퐵
## ௔,௧
## =
## 퐵
## ௔,௧
The term in
LinTS:
Equivalent,  in
## Lin-
## 휺−푮풓풆풆풅풚
## :
## 푓
## ௔,௧
## =
## 푓
## ௔,௧
## 푏
## ௜
## (푡)
## =
## 푥
## ௧
## 휇̂

## =       휃
## ෠
## ௔,௧
## =       퐴
## ௔,௧
## =
## Information Matrix
The term in
LinUCB:
## =     퐵
## ௔,௧
## =
## Covariance Matrix
## =       푥
## ௧
## =
User context
## =
## 휃
## ෠
## ௔,௧
## =퐿푖푛푒푎푟 푚표푑푒푙 푝푎푟푎푚푒푡푒푟푠
In whichcases did
wetrythisarm?
In whichcases
didthisarm get
good rewards?
In whichcase are
weright now?

## 60
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Questions ?
Contextual Multi-Armed Bandits (CMABs)

## 61
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## 1. Prologue
- Introduction to ReinforcementLearning
- Multi-ArmedBandits (MABs)
- ContextualMulti-ArmedBandits (CMABs)
5.MABs/CMABsExtensions
## 6. Projects
## 7. Course Conclusion
Contextual Multi-Armed Bandits (CMABs)
## Syllabus

## 62
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
A list of extensions
MABs/CMABs Extensions
## ➢
Top-KArmsSelection[HJ17]:TheobjectiveistoidentifytheKbest-performingarmsoutofalarger
set.Unliketraditionalbanditswhereweexploreallarmstoidentifytheoptimalaction,herewefocuson
identifying the top K arms with high confidence. This scenario arises in scenarios like personalized
recommendations (selecting the top K products for a user) or clinical trials (identifying the most
effectivetreatments).
[HJ17]H. Jiang, J. Li and M. Qiao. “Practical Algorithms for Best-K Identification in Multi-Armed Bandits”, ”. arXiv, 2017.
[WKA15]Z. Wen et al. “Efficient Learning in large scale combinatorial semi-bandits”. ICML, 2015.
## ➢
CombinatorialBandits[WKA15]:Combinatorialbanditsinvolveselectingasetofarms(anaction)
ratherthanjustasinglearm.Eachactioncorrespondstoacombinationofarms.Thelearner’sgoalisto
choose actions that maximize the combined rewards of the selected arms (instead of their individual
reward). This setting is particularly relevant when the arms are related or interact with each other.
Applications include resource allocation (e.g., distributing a fixed budget across advertising channels)
andrecommendationsystems(selectingabundleofitemsforauser).

## 63
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
A list of extensions
MABs/CMABs Extensions
## ➢
Bandits with delayed feedback [PB18]: In scenarios with delayed feedback, the agent does not
immediatelyobservetherewardafterpullinganarm.Instead,there’sadelay(severaliterations)before
receivingfeedback.Thisdelayintroducesadditionalchallengesbecausetheagentmustdecidewithout
knowing the immediate outcome. Applications include scenarios where decisions have consequences
beyondtheimmediatemomentlikeallocatingalimitedbudgetorchoosingmedicaltreatments.
[PB18]C. Pike-Burke, S. Agrawal, C. Szepesvari, S. Grünewälder. “Bandits with Delayed, Aggregated Anonymous Feedback”, Proceedings of Machine Learning Research2018.
[LKC19]A. Luedkte, E. Kaufmann and A.Chambaz“Asymptotically optimal algorithms for budgeted multiple play bandits”. Machine Learning Journal, 2019.
## ➢
Budget-Constrained Bandits [LKC19]: In budget-constrained multi-armed bandits, the decision-
makerhasalimitedbudgettoallocateacrossarms.Thegoalistomaximizerewardswhilestayingwithin
thebudgetconstraint.Imagineascenariowhereyouhaveafixedbudgetforexperimentation(e.g.,in
clinical trials or A/B testing). You need to allocate this budget wisely to explore different arms and
exploit the best-performing ones. Budget constraints add an extra layer of complexity to the bandit
problem,requiringefficientallocationstrategies.

## 64
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
A list of extensions
MABs/CMABs Extensions
## ➢
Volatile Arms [CX18]: Volatile arms refer to arms whose availability change over time. These arms
canbecomeavailableorunavailableineachround,makingtheproblemmoredynamicandchallenging.
Forexample,considere-commercescenariowhereproposedproductsandtheirstocksvaryovertime.
## Subclassesofthisproblemincludesleepingarmsbanditsandinfinitearmsbandits.
[CX18]Chen Lixing, Xu Jie and Lu Zhuo. “Contextual Combinatorial Multi-armed Bandits with Volatile Arms and Submodular Reward”, NeurIPS2018
[LLZ10]H. Liu, K. Liu and Q.Zhao. “Learning in a changing world: Non-Bayesian restless multi-armed bandit”. arXiv, 2010.
## ➢
RestlessBandits[LLZ10]:Intraditionalmulti-armedproblems,therewarddistributionofthearmsis
consideredfrozenwhennoactionisperformed.Howeverinreal-worldscenarios,forexamplecontent
recommendation,therewarddistributionisoftennotstationary,forexamplefollowingtrends.Under
therestlessbanditssetting,itisconsideredthattherewardsexpectationofarmsmaychangedynamically
evenfornotchosenarmsandalgorithmsaimatcapturingthisevolution.

## 65
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
A list of extensions
MABs/CMABs Extensions
## ➢
Multi-objective Bandits [DN13]: Multi-objective bandits involve optimizing multiple objectives
simultaneously. For instance, a decision-maker may want to maximize both revenue and customer
satisfaction.Theseobjectivesmayconflict,leadingtoanothertrade-offbetweentheobjectivesbesides
theusualoneforexploitation-exploration.Applicationsincludepersonalizedrecommendations,where
thegoalistooptimizemultipleuserpreferences(e.g.,relevance,diversity,novelty).
[DN13]M. M.Druganand A. Nowé. “Designing multi-objective multi-armed banditsalgorithms: a study”, IJCNN, 2013
[FL19]F. Liu and N. Shroff. “Data Poisoning Attacks on Stochastic Bandits”, arXiv, 2019
## ➢
PoisoningAttacksonBandits[FL19]:Stochasticmulti-armedbandits(S-MABs)arewidelyusedin
onlinerecommendationsystems,adaptivemedicaltreatment,andmore.Hence,attackersseektohijack
the behavior of bandit algorithms, causing catastrophic losses in real-world applications. For offline
attacks,theattackermay manipulate rewardsinhistoricaldatato forcethebanditalgorithmtopulla
specific arm with high probability. For online attacks, the attacker may use fake profiles to foster a
specificaction.

## 66
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
A list of extensions
MABs/CMABs Extensions
Severalextensionscanberelatedandthereexistmanyothers.Asharedfeatureisthattheyallare
motivated by real-world applications. Understanding the specificities of the real problem, the
relatedsettingsandimplementingalgorithmsthatwillbeabletohandlemostofthemiswhat
willmakethedifferenceafterdeployment.
## Imageextractedfrom:http://www.primarydigit.com/blog/archives/12-2015

## 67
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Questions ?
MABs/CMABs Extensions

## 68
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## 1. Prologue
- Introduction to ReinforcementLearning
- Multi-ArmedBandits (MABs)
- ContextualMulti-ArmedBandits (CMABs)
- MABs extensions
6.Projects
## 7. Course Conclusion
Contextual Multi-Armed Bandits (CMABs)
## Syllabus

## 69
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Short Projects Presentation:
MABs/CMABs extensions for real-world setting
## Projects
## Presentation

## 70
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Short Projects Presentation: Objectives
## Projects
## Generalfeatures:
## 
## Studentsworkbyteamof3.
## 
The group choose one extension (or more) to work on based on their preference (can be a
settingnotshowninclass).Studentsexplorethesettingandupdatetheirsimulatortoinclude
itbeforeimplementinganalgorithmtohandleit.
## 
## Asforthecourse,thoughitisaproject,theuseofchatgpt&coandexistingcodetosolvethe
problemisprohibited.
## 
Eachgroupsubmissionwillincludethecompletesoftware(withdataset)andaprojectreport.
## Submissionareduefor:27/03/2026?23h59(2weeksafterlastsession),

## 71
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Short Projects Presentation: Objectives
## Projects
## Missions:
## 
IdentifyandstudyanextensionofMABproblem.Relatedresearchpapersmustbeconsideredandcited.
## 
Updatethesimulatortomodelizethenewsettingintoit,anewdatasetmaybeintegratedintoitifdeemed
necessary.Theupdatemustbedescribedinthereport.
## 
Implementa MAB/CMAB algorithm taking intoaccountthe newsetting. Try toupdatethe algorithms
studiedinclassforthisnewsetting.
## 
Compare the performances of your previous MAB/CMAB algorithms (average over 10 runs) with your
newmethodsunderthissettingandanalyzetheminthereport.
## 
The code should be of quality, following PEP-8 standards, well-documented and modular. Report must
include workload distribution, justify all previous choices, explain the new setting and how the code
implementandhandleit.

## 72
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Short Projects Presentation: Objectives
## Projects
ProjectEvaluation:
## 
1 submission per group withthe complete software (with datasets)and the project report.Submission
areduefor:27/03/202623h59,evaluatedaccordingtothefollowing:
## 
KeepinmindGenAIisprohibitedforthiscourse,withsameconsequencesasanycheatmethod.A
projectdefensewillbeholdonlastsessionwithsimilarcriteria(10’+questions).

## 73
RL V2025 –IR4 IA S8 -ESAIP, 2025-26 –A.Letard
## Questions ?
## Projects