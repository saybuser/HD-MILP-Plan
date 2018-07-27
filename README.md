# HD-MILP-Plan

Hybrid Deep MILP Planner (HD-MILP-Plan) is a two-stage planner based on the learning and planning framework [1] that (i) learns the state transition function T(s<sub>t</sub>,a<sub>t</sub>) = s<sub>t+1</sub> of a factored [2] planning problem using Densely Connected Neural Networks [3] from data, and (ii) compiles multiple copies of the learned transition function T'(...T'(T'(T'(I,a<sub>0</sub>),a<sub>1</sub>),a<sub>2</sub>)...) = G (as visualized by Figure 1) into MILP and solves it using off-the-shelf MILP solver [4]. HD-MILP-Plan can handle discrete/continuous action spaces and continuous state spaces, arbitrarily complex state transition functions, linear constraints on actions/states and linear reward functions.

![alt text](./hdmilpplan.png)
Figure 1: Visualization of the learning and planning framework presented in [1] where red circles represent action variables, blue circles represent state variables, gray circles represent the activation units and w's represent the weights of the neural network.

## Improvements

HD-MILP-Plan

i) includes parsers for domain files that read in linear expressions of form: sum<sub>1..i..n</sub> x<sub>i</sub> ? k where ? can be <=, >= or ==. See translation folder for more details.

ii) handles goal (state) constraints.

## Dependencies

i) Data collection (input to training DNN): Data is collected using the RDDL-based domain simulator [5]. 

ii) Training DNN: The toolkit [6] is used to train DNNs. The final training parameters were recorded into dnn.txt file.

iii) Solver: Any off-the-shelf MILP solver works. In our paper [1], we used CPLEX solver [4].

For i) any domain simulator and for ii) any DNN training toolkit works. Example dnn.txt and domain files (under translation folder) are provided for navigation, hvac and reservoir domains. Therefore to run the planner, you will only need iii).

## Running HD-MILP-Plan

hd_milp_plan.py -d domain -i instance -h horizon -b bound -s sparsify (optional, set to 0.0 by default)

Example: python hd_milp_plan.py -d navigation -i 10x10 -h 10 -b True

## Verification Task

HD-MILP-Plan can also be used to verify different properties of DNNs by setting horizon -h to 1.

## Citation

If you are using HD-MILP-Plan, please cite the paper [1].

## References
[1] Buser Say, Ga Wu, Yu Qing Zhou, and Scott Sanner. [Nonlinear hybrid planning with deep net learned transition models and mixed-integer linear programming](http://static.ijcai.org/proceedings-2017/0104.pdf). In 26th IJCAI, pages 750–756, 2017.

[2] Craig Boutilier, Thomas Dean, and Steve Hanks. Decision-theoretic planning: Structural assumptions and computational leverage. JAIR, 11(1):1–94, 1999.

[3] Gao Huang, Zhuang Liu, and Kilian Weinberger. Densely connected convolutional networks. 2016.

[4] IBM ILOG CPLEX Optimization Studio CPLEX User's Manual, 2017.

[5] Scott Sanner. Relational dynamic influence diagram language (rddl): Language description. 2010.

[6] Ga Wu. [Github repository](https://github.com/wuga214/PAPER_IJCAI17_HybridPlanning_NeuralNetwork_MILP).
