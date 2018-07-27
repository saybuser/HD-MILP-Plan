# FD-BLP-Plan

Factored Deep BLP Planner (FD-BLP-Plan) [1] is a two-stage planner based on the learning and planning framework [2] that (i) learns the state transition function T(s<sub>t</sub>,a<sub>t</sub>) = s<sub>t+1</sub> of a factored [3] planning problem using Binarized Neural Networks [4] from data, and (ii) compiles multiple copies of the learned transition function T'(...T'(T'(T'(I,a<sub>0</sub>),a<sub>1</sub>),a<sub>2</sub>)...) = G (as visualized by Figure 1) into BLP and solves it using off-the-shelf BLP solver [5]. FD-BLP-Plan can handle both discrete and continuous action/state spaces, arbitrarily complex state transition functions, pseudo-boolean constraints on actions/states and pseudo-boolean reward functions.

![alt text](./hdmilpplan.png)
Figure 1: Visualization of the learning and planning framework presented in [2] where red circles represent action variables, blue circles represent state variables, gray circles represent the activation units and w's represent the weights of the neural network.

## Improvements

FD-BLP-Plan

i) includes parsers for domain files that read in pseudo-boolean expressions of form: sum<sub>1..i..n</sub> x<sub>i</sub> <= k. See translation folder for more details.

ii) handles reward functions.

iii) can make use of known transition functions (i.e., the transition function for a subset of state variables can be provided as input - see Inventory Control example).

## Dependencies

i) Data collection (input to training BNN [4]): Data is collected using the RDDL-based domain simulator [6]. 

ii) Training BNN: The toolkit [7] is used to train BNNs. The final training parameters were recorded into bnn.txt and normalization.txt files.

iii) Solver: Any off-the-shelf BLP solver works. In our paper [1], we used CPLEX solver [5].

For i) any domain simulator and for ii) any BNN training toolkit works. Example bnn.txt, normalization.txt and domain files (under translation folder) are provided for navigation, inventory and sysadmin domains. Therefore to run the planner, you will only need iii).

## Running FD-BLP-Plan

fd_blp_plan.py -d domain -i instance -h horizon -o optimize

Example: python fd_blp_plan.py -d navigation -i 3x3 -h 4 -o False

## Verification Task

FD-BLP-Plan can also be used to verify different properties of BNNs by setting horizon -h to 1.

## Known Limitations

i) Input files in translation folder only accepts pseudo-boolean constraints/expressions in the form of: sum<sub>1..i..n</sub> x<sub>i</sub> ? k where ? can be <=, >= or ==.

## Citation

If you are using FD-BLP-Plan, please cite the papers [1,2].

## References
[1] Buser Say and Scott Sanner. [Planning in Factored State and Action Spaces with Learned Binarized Neural Network Transition Models](https://www.ijcai.org/proceedings/2018/0669.pdf). In 27th IJCAI, pages 4815-4821, 2018.

[2] Buser Say, Ga Wu, Yu Qing Zhou, and Scott Sanner. [Nonlinear hybrid planning with deep net learned transition models and mixed-integer linear programming](http://static.ijcai.org/proceedings-2017/0104.pdf). In 26th IJCAI, pages 750–756, 2017.

[3] Craig Boutilier, Thomas Dean, and Steve Hanks. Decision-theoretic planning: Structural assumptions and computational leverage. JAIR, 11(1):1–94, 1999.

[4] Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized neural networks. In 29th NIPS, pages 4107–4115. Curran Associates, Inc., 2016.

[5] IBM ILOG CPLEX Optimization Studio CPLEX User's Manual, 2017.

[6] Scott Sanner. Relational dynamic influence diagram language (rddl): Language description. 2010.

[7] Matthieu Courbariaux. BinaryNet. https://github.com/MatthieuCourbariaux/BinaryNet
