// =============================================
//  BITS WILP EC3 Prep Hub — Mock Test Engine
//  Full Bank: 160 questions (40 per subject)
// =============================================

const questionBank = {

  // ================================================
  // MFML — 40 Questions (14 Easy, 14 Medium, 12 Hard)
  // ================================================
  MFML: {
    easy: [
      { q:"Which property defines a symmetric matrix?", options:["A = Aᵀ","A = -Aᵀ","A has no eigenvalues","A is always invertible"], answer:0, explanation:"A symmetric matrix satisfies A = Aᵀ. All real symmetric matrices have real eigenvalues and orthogonal eigenvectors." },
      { q:"The dot product of two orthogonal vectors is:", options:["1","-1","0","Undefined"], answer:2, explanation:"Orthogonal vectors are perpendicular: u·v = |u||v|cos(90°) = 0." },
      { q:"What is the rank of a zero matrix?", options:["0","1","n","Undefined"], answer:0, explanation:"A zero matrix has no linearly independent rows/columns, so rank = 0." },
      { q:"If eigenvalue of A is λ, what is the eigenvalue of A²?", options:["2λ","λ²","√λ","1/λ"], answer:1, explanation:"If Av = λv, then A²v = A(λv) = λ²v. Eigenvalue of A² is λ²." },
      { q:"The trace of a matrix equals:", options:["Sum of all elements","Sum of diagonal elements","Product of eigenvalues","Determinant"], answer:1, explanation:"tr(A) = Σaᵢᵢ = sum of diagonal elements. Also equals sum of all eigenvalues." },
      { q:"det(Aᵀ) equals:", options:["det(A)²","1/det(A)","det(A)","-det(A)"], answer:2, explanation:"det(Aᵀ) = det(A). Transposing a matrix does not change its determinant." },
      { q:"The L2 norm of vector v = [3, 4] is:", options:["7","5","25","3.5"], answer:1, explanation:"||v||₂ = √(3²+4²) = √25 = 5. This is the Euclidean length." },
      { q:"The gradient ∇f points in the direction of:", options:["Steepest descent","Steepest ascent","Minimum value","Zero change"], answer:1, explanation:"∇f points toward steepest ascent. Gradient descent uses -∇f to go downhill." },
      { q:"A vector space must be closed under:", options:["Matrix multiplication","Vector addition and scalar multiplication","Division","Exponentiation"], answer:1, explanation:"Closure under addition and scalar multiplication is required by vector space axioms." },
      { q:"[1,2] and [2,4] are:", options:["Orthogonal","Linearly independent","Linearly dependent","Orthonormal"], answer:2, explanation:"[2,4] = 2×[1,2], so one is a scalar multiple of the other — linearly dependent." },
      { q:"An orthonormal basis consists of vectors that are:", options:["Parallel and unit length","Orthogonal and unit length","Orthogonal and any length","Parallel and any length"], answer:1, explanation:"Orthonormal = pairwise orthogonal AND each has unit length (||v||=1)." },
      { q:"The null space of A contains all x such that:", options:["Ax = b","Ax = 0","Ax = I","Ax = Aᵀ"], answer:1, explanation:"Null space (kernel) = {x : Ax = 0}. Its dimension is the nullity of A." },
      { q:"For a square matrix A, det(A) = 0 means A is:", options:["Positive definite","Invertible","Singular (no inverse)","Symmetric"], answer:2, explanation:"det(A) = 0 means A is singular — no inverse exists." },
      { q:"The Cauchy-Schwarz inequality states |u·v| is:", options:["≤ ||u|| ||v||","≥ ||u|| ||v||","= ||u|| + ||v||","= ||u|| × ||v||"], answer:0, explanation:"|u·v| ≤ ||u|| ||v||. Equality holds iff u and v are parallel." }
    ],
    medium: [
      { q:"Gradient of f(x,y) = x²y + y³ at (1,2):", options:["[4,13]","[4,12]","[2,13]","[4,11]"], answer:0, explanation:"∂f/∂x = 2xy = 4, ∂f/∂y = x²+3y² = 1+12 = 13. So ∇f = [4,13]." },
      { q:"A matrix is positive definite iff:", options:["All eigenvalues > 0","All diagonal entries > 0","det > 0","All entries > 0"], answer:0, explanation:"Positive definite means all eigenvalues > 0, equivalently xᵀAx > 0 for all non-zero x." },
      { q:"SVD decomposes matrix A as:", options:["LDLᵀ","QR","UΣVᵀ","PDP⁻¹"], answer:2, explanation:"A = UΣVᵀ. U = left singular vectors, Σ = singular values, V = right singular vectors." },
      { q:"Rank-Nullity theorem: for A in R(m×n):", options:["rank+nullity = m","rank+nullity = n","rank × nullity = mn","rank = nullity"], answer:1, explanation:"rank(A) + nullity(A) = n (number of columns)." },
      { q:"The Frobenius norm ||A||_F equals:", options:["Sum of aij","sqrt(sum of aij squared)","max|aij|","Sum of diagonal"], answer:1, explanation:"||A||_F = sqrt(sum of all aij²) = sqrt(tr(AᵀA))." },
      { q:"Best rank-k approximation (Eckart-Young) is:", options:["Sum of top k singular triplets","A minus last k columns","First k rows of A","Uk*Sigma_k only"], answer:0, explanation:"Ak = sum of top k terms σᵢuᵢvᵢᵀ. Minimises ||A-Ak||_F." },
      { q:"Hessian of f(x,y) = x² + 2xy + y² is:", options:["[[2,2],[2,2]]","[[2,1],[1,2]]","[[2,0],[0,2]]","[[1,2],[2,1]]"], answer:0, explanation:"H11=2, H12=H21=2, H22=2. Hessian = [[2,2],[2,2]]." },
      { q:"Projection of b onto column space of A is:", options:["A(AᵀA)⁻¹Aᵀb","AᵀAb","(AᵀA)⁻¹b","AAᵀb"], answer:0, explanation:"Projection matrix P = A(AᵀA)⁻¹Aᵀ. Pb gives the projection." },
      { q:"Gram-Schmidt orthogonalisation produces:", options:["Parallel vectors","Eigenvectors","Orthogonal vectors spanning same space","Diagonal matrix"], answer:2, explanation:"Gram-Schmidt converts linearly independent vectors to orthogonal ones with same span." },
      { q:"For convex function f, every local minimum is:", options:["A saddle point","Also a global minimum","Unique","On the boundary"], answer:1, explanation:"Convexity guarantees any local minimum is also a global minimum." },
      { q:"The chain rule for f(g(x)) gives df/dx as:", options:["df/dg + dg/dx","df/dg × dg/dx","df/dx × dg/dx","df/dg / dg/dx"], answer:1, explanation:"df/dx = (df/dg)(dg/dx). Basis of backpropagation." },
      { q:"Eigendecomposition A = PDP⁻¹ requires A to have:", options:["All positive eigenvalues","n linearly independent eigenvectors","Symmetric structure","det = 1"], answer:1, explanation:"Eigendecomposition exists iff A has n linearly independent eigenvectors." },
      { q:"Condition number κ(A) = σ_max/σ_min measures:", options:["How close to singular","Number of eigenvalues","Determinant magnitude","Trace value"], answer:0, explanation:"High condition number means near-singular and ill-conditioned." },
      { q:"Linear regression solution β=(XᵀX)⁻¹Xᵀy is valid when:", options:["X has full column rank","X is square","y is binary","X has more columns than rows"], answer:0, explanation:"XᵀX invertible iff X has full column rank (no multicollinearity)." }
    ],
    hard: [
      { q:"For m×n matrix with m<n, null space has dimension:", options:["0","≥ n-m","Exactly m","Undefined"], answer:1, explanation:"nullity = n - rank(A) ≥ n - m > 0. A wide matrix always has a non-trivial null space." },
      { q:"KKT conditions include all of:", options:["Stationarity, primal/dual feasibility, complementary slackness","Gradient = 0 only","Positive definite Hessian","Convex objective only"], answer:0, explanation:"KKT: stationarity, primal feasibility, dual feasibility (λ≥0), and complementary slackness (λᵢgᵢ=0)." },
      { q:"For L-smooth function, gradient descent converges when step size α satisfies:", options:["α < 2/L","α = L","α > L","α = 1"], answer:0, explanation:"For L-smooth f, GD converges with α < 2/L. Optimal step is α=1/L." },
      { q:"Moore-Penrose pseudoinverse A+b gives:", options:["Maximum norm solution","Minimum norm least-squares solution","Exact solution always","Maximum residual solution"], answer:1, explanation:"A+b = minimum-norm least-squares solution. Computed via SVD: A+ = VΣ+Uᵀ." },
      { q:"Spectral theorem: real symmetric A writes as:", options:["A=LU","A=QΛQᵀ, Q orthogonal, Λ real diagonal","A=UΣVᵀ with U≠V","A=PDP⁻¹, P not orthogonal"], answer:1, explanation:"Spectral theorem: A=QΛQᵀ. Q's columns are orthonormal eigenvectors, Λ has real eigenvalues." },
      { q:"PCA principal components are:", options:["Rows of data matrix","Eigenvectors of covariance sorted by decreasing eigenvalue","Columns of data matrix","Left singular vectors sorted ascending"], answer:1, explanation:"PCA: eigenvectors of covariance matrix sorted by decreasing eigenvalue." },
      { q:"Second-order Taylor expansion of f(x) around x₀:", options:["f(x₀)+∇f·δ","f(x₀)+∇f·δ + ½δᵀH·δ","½H·δ²","∇f·δ+H"], answer:1, explanation:"f(x) ≈ f(x₀)+∇f(x₀)ᵀδ+½δᵀH(x₀)δ where δ=x-x₀." },
      { q:"If A has eigenvalues λ₁,...,λₙ, eᴬ has eigenvalues:", options:["exp(λ₁),...,exp(λₙ)","e+λᵢ","λᵢ²","1/λᵢ"], answer:0, explanation:"If Av=λv then eᴬv=eˡv. Follows from series expansion." },
      { q:"Nuclear norm (low-rank regularisation) is:", options:["Sum of |eigenvalues|","Sum of singular values","Frobenius norm squared","Largest singular value"], answer:1, explanation:"||A||* = Σσᵢ. Convex envelope of rank function, promotes low-rank solutions." },
      { q:"MLE estimate of Gaussian covariance is:", options:["(1/n)Σ(xᵢ-μ)(xᵢ-μ)ᵀ","(1/(n-1))Σ(xᵢ-μ)(xᵢ-μ)ᵀ","Sum of xᵢ²","Identity matrix"], answer:0, explanation:"MLE: (1/n)Σ — biased. Unbiased uses 1/(n-1) (Bessel's correction)." },
      { q:"Jacobian of f: Rⁿ→Rᵐ has shape:", options:["n×m","m×n","n×n","m×m"], answer:1, explanation:"J ∈ Rᵐˣⁿ with Jᵢⱼ=∂fᵢ/∂xⱼ." },
      { q:"Strong duality in convex optimisation holds when:", options:["Primal is unbounded","Slater's condition holds","λ=0","f is linear"], answer:1, explanation:"Slater's condition: exists strictly feasible point → strong duality (zero duality gap)." }
    ]
  },

  // ================================================
  // ISM — 40 Questions (14 Easy, 14 Medium, 12 Hard)
  // ================================================
  ISM: {
    easy: [
      { q:"The mean of {2, 4, 6, 8, 10} is:", options:["5","6","4","7"], answer:1, explanation:"Mean = (2+4+6+8+10)/5 = 30/5 = 6." },
      { q:"Standard deviation is the square root of:", options:["Mean","Variance","Covariance","Correlation"], answer:1, explanation:"σ = √Var(X). SD is in the same units as the data." },
      { q:"P(A ∪ B) = P(A) + P(B) when A and B are:", options:["Independent","Mutually exclusive","Complementary","Correlated"], answer:1, explanation:"Mutually exclusive: P(A∩B)=0, so P(A∪B)=P(A)+P(B)." },
      { q:"For X ~ U(0,1), E[X] =", options:["0","1","0.5","0.25"], answer:2, explanation:"For U(a,b), E[X] = (a+b)/2 = 0.5." },
      { q:"Bayes' theorem calculates:", options:["P(A) from P(B)","P(A|B) using P(B|A) and prior P(A)","P(A ∩ B)","P(A ∪ B)"], answer:1, explanation:"Bayes: P(A|B) = P(B|A)P(A)/P(B). Updates prior to posterior given evidence." },
      { q:"In normal distribution, ~95% data lies within:", options:["1 SD","2 SD","3 SD","0.5 SD"], answer:1, explanation:"68-95-99.7 rule: 95% within ±2σ. Key exam fact!" },
      { q:"Median of {3, 7, 5, 1, 9} is:", options:["5","7","3","4.5"], answer:0, explanation:"Sort: {1,3,5,7,9}. Middle value = 5." },
      { q:"Cov(X,X) equals:", options:["0","1","Var(X)","Std(X)"], answer:2, explanation:"Cov(X,X) = E[(X-μ)²] = Var(X)." },
      { q:"If X ~ N(μ, σ²), standardising gives Z =", options:["X/σ","(X-μ)/σ","X-μ","(X-σ)/μ"], answer:1, explanation:"Z = (X-μ)/σ ~ N(0,1)." },
      { q:"Correlation coefficient r ranges from:", options:["0 to 1","-1 to 1","0 to ∞","-∞ to ∞"], answer:1, explanation:"r = Cov(X,Y)/(σxσy) ∈ [-1,1]." },
      { q:"The mode is:", options:["Average value","Middle value","Most frequent value","Smallest value"], answer:2, explanation:"Mode = most frequently occurring value." },
      { q:"For independent events A and B, P(A ∩ B) =", options:["P(A) + P(B)","P(A) × P(B)","P(A) - P(B)","P(A)/P(B)"], answer:1, explanation:"Independence: P(A∩B) = P(A)P(B)." },
      { q:"E[c] for constant c is:", options:["0","c","c²","Undefined"], answer:1, explanation:"E[c] = c. Constants are deterministic." },
      { q:"Var(c) for constant c is:", options:["c","c²","0","1"], answer:2, explanation:"Var(c) = 0. Constants have no variability." }
    ],
    medium: [
      { q:"Bias-Variance tradeoff: total expected error =", options:["Bias + Variance","Bias² + Variance + Noise","Bias × Variance","Bias² × Variance"], answer:1, explanation:"E[(y-ŷ)²] = Bias² + Variance + Irreducible Noise." },
      { q:"MLE finds parameters that:", options:["Minimise variance","Maximise P(data|parameters)","Minimise bias","Maximise the prior"], answer:1, explanation:"MLE: θ̂ = argmax P(data|θ)." },
      { q:"Central Limit Theorem: sample mean X̄ of large n is approximately:", options:["Uniform","Binomial","Normal","Exponential"], answer:2, explanation:"CLT: X̄ ~ N(μ, σ²/n) for large n, regardless of population distribution." },
      { q:"95% confidence interval means:", options:["P(θ in interval) = 0.95","95% of such intervals contain θ","Estimate is 95% accurate","Error < 5%"], answer:1, explanation:"Frequentist CI: 95% of intervals constructed this way contain the true parameter." },
      { q:"Type I error (α) is:", options:["False negative","False positive — reject true H₀","Accept false H₀","Correct rejection"], answer:1, explanation:"Type I: reject H₀ when it's true. Type II (β): fail to reject false H₀." },
      { q:"p-value is:", options:["P(H₀ is true)","P(data at least this extreme under H₀)","P(H₁ is true)","Effect size"], answer:1, explanation:"p-value = P(T ≥ t_obs | H₀). Small p → reject H₀." },
      { q:"Pearson's r measures:", options:["Any monotonic relationship","Linear relationship strength","Rank correlation","Causation"], answer:1, explanation:"Pearson's r measures linear association. For monotonic non-linear, use Spearman's ρ." },
      { q:"Bayesian posterior is proportional to:", options:["Prior only","Likelihood only","Prior × Likelihood","Likelihood / Prior"], answer:2, explanation:"P(θ|data) ∝ P(data|θ) × P(θ)." },
      { q:"t-distribution compared to N(0,1) has:", options:["Always heavier tails","Lower tails","Same shape","Skewness"], answer:0, explanation:"t-distribution has heavier tails. As df → ∞, t → N(0,1)." },
      { q:"Overfitting is indicated by:", options:["High train error, high test error","Low train error, high test error","Low train error, low test error","High train error, low test error"], answer:1, explanation:"Overfitting: model memorises training data → low training error, high test error." },
      { q:"Chi-squared test is used for:", options:["Mean comparison","Goodness of fit / independence of categorical variables","Regression","ANOVA"], answer:1, explanation:"χ² test: goodness-of-fit or independence of two categorical variables." },
      { q:"Law of total probability: P(A) =", options:["Σ P(A|Bᵢ)P(Bᵢ)","Σ P(A∩Bᵢ)","Σ P(Bᵢ|A)","P(A)P(B)"], answer:0, explanation:"P(A) = Σᵢ P(A|Bᵢ)P(Bᵢ) over a partition {Bᵢ}." },
      { q:"F-test in ANOVA compares:", options:["Two means","Between-group vs within-group variance","Two variances only","Regression coefficients"], answer:1, explanation:"F = Between-group variance / Within-group variance. Large F → groups differ." },
      { q:"Regularisation primarily prevents:", options:["Underfitting","Overfitting by penalising complexity","Bias","Missing data"], answer:1, explanation:"L1/L2 regularisation adds complexity penalty → reduces overfitting." }
    ],
    hard: [
      { q:"Cramer-Rao bound gives minimum variance of:", options:["Any estimator","Any unbiased estimator","MLE only","Bayesian estimator"], answer:1, explanation:"CRLB: Var(θ̂) ≥ 1/I(θ) for any unbiased estimator. MLE is asymptotically efficient." },
      { q:"Sufficient statistic T(X) means:", options:["T(X) equals θ","P(X|T,θ) independent of θ","T(X) minimises MSE","T(X) is Gaussian"], answer:1, explanation:"Sufficiency: T captures all info about θ. Fisher-Neyman: f(x|θ) = g(T(x),θ)h(x)." },
      { q:"EM algorithm is used when:", options:["Data is complete","There are latent/hidden variables","MLE has closed form","Data is Gaussian"], answer:1, explanation:"EM: E-step computes expected log-likelihood, M-step maximises it. Handles latent variables." },
      { q:"AIC = 2k - 2ln(L̂) penalises:", options:["Sample size","Number of parameters k","Bias","Variance only"], answer:1, explanation:"AIC penalises model complexity k. Lower AIC = better model." },
      { q:"Bootstrap estimates uncertainty by:", options:["Analytical formulas","Resampling with replacement from observed data","Cross-validation","Bayesian priors"], answer:1, explanation:"Bootstrap: resample n observations with replacement B times, compute statistic each time." },
      { q:"Power of a test is:", options:["P(reject H₀ | H₀ true)","P(reject H₀ | H₀ false)","1 - p-value","α + β"], answer:1, explanation:"Power = 1-β = P(correctly reject false H₀). Target ≥ 0.8." },
      { q:"KL divergence KL(P||Q) is:", options:["Symmetric distance","How much P diverges from Q (not symmetric)","Variance of P","Covariance of P and Q"], answer:1, explanation:"KL(P||Q) = Σ P(x)log(P(x)/Q(x)) ≥ 0. Not symmetric. Used in variational inference." },
      { q:"MCMC is used to:", options:["Compute exact integrals","Sample from complex posterior distributions","Compute MLE analytically","Cross-validate"], answer:1, explanation:"MCMC (Gibbs, MH) generates samples from P(θ|data) when intractable." },
      { q:"Regression to the mean implies:", options:["Predictions always equal mean","Extreme observations tend to be followed by less extreme ones","Residuals are zero","Slope > 1"], answer:1, explanation:"Extreme scores tend to be closer to mean on next measurement. Galton discovered this." },
      { q:"Complete statistic T has the property:", options:["T captures all info","E[g(T)] = 0 for all θ implies g ≡ 0","T achieves CRLB","T is always sufficient"], answer:1, explanation:"Completeness + sufficiency → UMVUE via Lehmann-Scheffé theorem." },
      { q:"Jeffreys prior is:", options:["Flat uniform prior","Invariant under reparameterisation","Equal to likelihood","Conjugate prior"], answer:1, explanation:"Jeffreys: p(θ) ∝ sqrt(det I(θ)). Invariant to parameter transformations." },
      { q:"Bonferroni correction adjusts α to:", options:["α × m","α / m","sqrt(α)","α²"], answer:1, explanation:"Bonferroni: use α/m per test for m tests. Controls family-wise error rate ≤ α." }
    ]
  },

  // ================================================
  // ML — 40 Questions (14 Easy, 14 Medium, 12 Hard)
  // ================================================
  ML: {
    easy: [
      { q:"Supervised learning requires:", options:["Unlabelled data","Labelled input-output pairs","Only input features","Reward signals"], answer:1, explanation:"Supervised: train on (X,y) pairs. Unsupervised: only X. RL: reward signals." },
      { q:"Overfitting means:", options:["Model too simple","Model memorises training but fails on new data","High bias","Low variance"], answer:1, explanation:"Overfitting: low training error, high validation/test error." },
      { q:"KNN classifies by:", options:["Decision trees","Majority vote of k nearest neighbours","Logistic regression","SVM margins"], answer:1, explanation:"KNN: find k closest training points, take majority class (or average for regression)." },
      { q:"Validation set is used to:", options:["Train the model","Tune hyperparameters","Evaluate final model","Augment training data"], answer:1, explanation:"Validation: used during training to select hyperparameters. Test set: final unbiased evaluation only." },
      { q:"Linear regression minimises:", options:["Absolute error","Sum of squared residuals (SSE)","Cross-entropy","Hinge loss"], answer:1, explanation:"OLS minimises SSE = Σ(yᵢ - ŷᵢ)². Closed form: β = (XᵀX)⁻¹Xᵀy." },
      { q:"Logistic regression is used for:", options:["Regression problems","Binary classification","Clustering","Dimensionality reduction"], answer:1, explanation:"Logistic regression outputs P(y=1|x) = σ(wᵀx+b). Decision boundary is linear." },
      { q:"Decision tree splits are chosen to maximise:", options:["Depth","Information gain or Gini impurity reduction","Node count","Leaf count"], answer:1, explanation:"At each node, find feature/threshold that maximally reduces Gini or entropy." },
      { q:"Cross-validation helps estimate:", options:["Training accuracy","Generalisation performance","Optimal learning rate","Number of features"], answer:1, explanation:"k-fold CV: average performance across k folds estimates generalisation." },
      { q:"Feature scaling is important for:", options:["Decision trees","Distance-based algorithms like KNN and SVM","Random forests","Naive Bayes"], answer:1, explanation:"KNN, SVM, gradient descent are sensitive to feature scales. Trees are invariant." },
      { q:"Sigmoid σ(z) maps z to:", options:["(-∞, +∞)","(0, 1)","(-1, 1)","[0, ∞)"], answer:1, explanation:"σ(z) = 1/(1+e⁻ᶻ) ∈ (0,1). Used for probability output in logistic regression." },
      { q:"Ensemble methods combine:", options:["Features","Multiple models to improve performance","Training and test sets","Network layers"], answer:1, explanation:"Ensemble: combine weak learners (bagging=parallel, boosting=sequential) → stronger model." },
      { q:"Random Forest uses which technique to decorrelate trees?", options:["Boosting","Feature subsampling + bootstrap","Pruning","Dropout"], answer:1, explanation:"RF: random feature subset per split + bootstrap sampling → decorrelated trees." },
      { q:"Gini impurity for a pure node is:", options:["1","0.5","0","Undefined"], answer:2, explanation:"Pure node: one class has p=1. Gini = 1-1² = 0." },
      { q:"Gradient descent update: θ :=", options:["θ - α × f(θ)","θ - α × ∇f(θ)","θ - ∇f(θ)/α","θ + α × ∇f(θ)"], answer:1, explanation:"θ := θ - α∇f(θ). Move opposite to gradient direction." }
    ],
    medium: [
      { q:"L1 regularisation (Lasso) produces:", options:["Ridge shrinkage","Sparse solutions (some weights exactly zero)","Smoother weights","No regularisation"], answer:1, explanation:"L1 penalty causes weights to snap to exactly zero → feature selection." },
      { q:"In SVM, the margin is:", options:["Distance from origin to hyperplane","Distance between support hyperplanes (2/||w||)","Largest eigenvalue","Sum of weights"], answer:1, explanation:"Margin = 2/||w||. SVM maximises this by minimising ||w||²." },
      { q:"Gradient Boosting builds trees:", options:["In parallel","Sequentially, each fitting residuals of previous","On random features only","Using bagging only"], answer:1, explanation:"GB: each tree fits pseudo-residuals. Trees added sequentially: F(x) += η×hₜ(x)." },
      { q:"The kernel trick in SVM allows:", options:["Reducing dimensions","Working in high-dim feature space without explicit mapping","Faster training","Removing support vectors"], answer:1, explanation:"Kernel K(xᵢ,xⱼ)=φ(xᵢ)ᵀφ(xⱼ) without computing φ. Common: RBF." },
      { q:"Naive Bayes assumes:", options:["Features are correlated","Features are conditionally independent given class","Gaussian likelihood always","Equal class priors"], answer:1, explanation:"NB: P(x|y) = ΠᵢP(xᵢ|y). Strong independence assumption." },
      { q:"VC dimension measures:", options:["Parameter count","Model capacity — max points it can shatter","Accuracy","Computational cost"], answer:1, explanation:"VC dim = max m such that model can shatter m points. Linear classifiers in Rⁿ: VC dim = n+1." },
      { q:"Increasing classifier threshold typically:", options:["Increases both precision and recall","Increases precision, decreases recall","Decreases precision, increases recall","Affects neither"], answer:1, explanation:"Higher threshold → fewer positives predicted → higher precision, lower recall." },
      { q:"AdaBoost reweights samples by:", options:["Uniform weights always","Increasing weight on misclassified samples","Decreasing all weights","Using gradient of loss"], answer:1, explanation:"AdaBoost: misclassified samples get higher weight → next learner focuses on hard examples." },
      { q:"Stochastic gradient descent (SGD) uses:", options:["All training data per update","One or mini-batch sample(s) per update","Batch size = epoch size","Exact gradient"], answer:1, explanation:"SGD updates per mini-batch. Noisy but faster than full batch GD." },
      { q:"ROC curve plots:", options:["Precision vs Recall","TPR vs FPR at various thresholds","Accuracy vs threshold","Loss vs epoch"], answer:1, explanation:"ROC: TPR vs FPR. AUC=0.5 → random, 1.0 → perfect." },
      { q:"PCA is used for:", options:["Classification","Dimensionality reduction and visualisation","Clustering only","Regression only"], answer:1, explanation:"PCA finds orthogonal directions of max variance for dimensionality reduction." },
      { q:"Complex models (deep decision trees) have:", options:["High bias, low variance","Low bias, high variance","High bias, high variance","Low bias, low variance"], answer:1, explanation:"Complex models: low bias (flexible) but high variance (sensitive to training data)." },
      { q:"K-means converges to:", options:["Global optimum always","Local optimum","Saddle point","Random assignment"], answer:1, explanation:"K-means guaranteed to converge but to local minimum. Run multiple times with different inits." },
      { q:"Elbow method in K-means selects:", options:["Learning rate","Optimal K","Distance metric","Feature count"], answer:1, explanation:"Plot inertia vs K; elbow = point of diminishing returns for adding more clusters." }
    ],
    hard: [
      { q:"PAC learning sample complexity scales as:", options:["O(1/ε)","O((1/ε)log(1/δ))","O(d²)","O(2^d)"], answer:1, explanation:"PAC: n = O((1/ε)(d + log(1/δ))) where ε=error, δ=failure prob, d=VC dim." },
      { q:"Rademacher complexity measures:", options:["Parameter count","Model capacity via fitting random ±1 labels","Training error","Regularisation penalty"], answer:1, explanation:"Rademacher complexity: ability to fit random labels. Smaller → better generalisation bounds." },
      { q:"XGBoost key improvement over vanilla GBM:", options:["Logistic loss","Second-order Taylor expansion + tree regularisation","Bootstrap sampling","Random features"], answer:1, explanation:"XGBoost: uses 2nd-order Taylor expansion of loss + regularisation term on tree structure." },
      { q:"Gaussian Process prediction variance represents:", options:["Training error","Epistemic (model) uncertainty","Noise variance","Bias"], answer:1, explanation:"GP posterior variance = uncertainty. Low near training points, high far away." },
      { q:"Representer theorem implies SVM solution lies in:", options:["Feature space","Span of kernel evaluations K(xᵢ, ·) on training data","Weight space only","Random subspace"], answer:1, explanation:"Representer theorem: optimal w = Σαᵢφ(xᵢ). Only training kernel evaluations needed." },
      { q:"SHAP values satisfy which axioms?", options:["Accuracy only","Efficiency, symmetry, dummy, additivity (Shapley axioms)","Maximise feature importance","Minimise complexity"], answer:1, explanation:"SHAP = Shapley values: unique attribution satisfying efficiency, symmetry, dummy, linearity." },
      { q:"Online learning regret is defined as:", options:["Test error","Cumulative loss vs best fixed hypothesis in hindsight","Gradient norm","Model complexity"], answer:1, explanation:"Regret = ΣLₜ(hₜ) - min_h ΣLₜ(h). Good algorithms achieve sublinear O(√T) regret." },
      { q:"Model calibration refers to:", options:["Hyperparameter tuning","Predicted probabilities matching true frequencies","Reducing model size","Feature scaling"], answer:1, explanation:"Calibrated model: P(y=1|ŷ=0.7) ≈ 0.7. Platt scaling can recalibrate models." },
      { q:"Counterfactual explanation answers:", options:["Why was prediction made?","What minimal input change would flip the prediction?","Feature importances?","Model accuracy?"], answer:1, explanation:"Counterfactual: 'Change X from a to b to flip prediction.' Actionable explanations." },
      { q:"Conformal prediction provides:", options:["Point estimates","Valid prediction sets with coverage guarantee","MAP estimates","Posterior distributions"], answer:1, explanation:"Conformal: set C(x) with P(y ∈ C(x)) ≥ 1-α. Distribution-free guarantee." },
      { q:"Causal inference differs from prediction because:", options:["Uses more data","Distinguishes intervention do(X=x) from conditioning P(Y|X=x)","Requires neural nets","Ignores confounders"], answer:1, explanation:"Causal: setting X ≠ conditioning on X. Pearl's do-calculus formalises interventions." },
      { q:"No Free Lunch theorem states:", options:["All algorithms same complexity","No single algorithm outperforms all others averaged over all problems","Ensembles always win","Linear models are best"], answer:1, explanation:"NFL: averaged over all data-generating distributions, all algorithms perform equally." }
    ]
  },

  // ================================================
  // DNN — 40 Questions (14 Easy, 14 Medium, 12 Hard)
  // ================================================
  DNN: {
    easy: [
      { q:"Output layer activation for binary classification:", options:["ReLU","Sigmoid","Tanh","Linear"], answer:1, explanation:"Sigmoid ∈ (0,1) gives probability. Multi-class: Softmax. Regression: linear." },
      { q:"Backpropagation computes gradients using:", options:["Forward pass only","Chain rule backwards through the network","Numerical approximation","Random perturbation"], answer:1, explanation:"Backprop: propagate gradient from loss backward using chain rule. Efficient O(W) computation." },
      { q:"Vanishing gradient causes:", options:["Faster training","Earlier layers learn very slowly","Overfitting","Higher accuracy"], answer:1, explanation:"Vanishing: gradients shrink through sigmoid/tanh layers → early layers barely update." },
      { q:"Dropout regularisation works by:", options:["Removing all neurons","Randomly zeroing fraction of neurons during training","Adding weight noise","Reducing learning rate"], answer:1, explanation:"Dropout(p): randomly zeros neurons with probability p. Prevents co-adaptation." },
      { q:"Batch normalisation normalises:", options:["Input data only","Activations within each mini-batch","Only final layer","Weights only"], answer:1, explanation:"BN: normalises activations to zero mean, unit variance per mini-batch, then learns γ,β scale/shift." },
      { q:"ReLU is defined as:", options:["1/(1+e⁻ˣ)","max(0, x)","tanh(x)","x²"], answer:1, explanation:"ReLU = max(0,x). Simple, avoids vanishing gradient for positive inputs. Most common activation." },
      { q:"CNNs are well-suited for:", options:["Time series only","Image and spatial data","Tabular data","Text only"], answer:1, explanation:"CNNs exploit local structure and translation invariance via local connectivity + weight sharing." },
      { q:"Softmax converts logits to:", options:["Binary values","Probability distribution summing to 1","Z-scores","Absolute values"], answer:1, explanation:"Softmax(zᵢ) = eᶻⁱ/Σeᶻʲ ∈ (0,1), sums to 1. Multi-class output layer." },
      { q:"RNNs are designed for:", options:["Static images","Sequential/temporal data","Tabular data","Clustering"], answer:1, explanation:"RNNs have hidden state for temporal dependencies. Good for time series and text." },
      { q:"Transfer learning reuses:", options:["Training data","Pre-trained model weights as starting point","Architecture only","Loss function"], answer:1, explanation:"Fine-tune pretrained model (e.g., ImageNet CNN) on new task with fewer data." },
      { q:"Parameters in FC layer (n inputs, m outputs):", options:["n + m","n×m + m (weights + biases)","n×m","2nm"], answer:1, explanation:"n×m weights + m biases = m(n+1) total parameters." },
      { q:"Which optimiser adapts learning rate per parameter?", options:["SGD","Adam","Momentum SGD","Vanilla GD"], answer:1, explanation:"Adam maintains per-parameter adaptive learning rates using first and second gradient moments." },
      { q:"Max pooling:", options:["Averages values","Takes maximum in pooling window","Multiplies by filter","Adds bias"], answer:1, explanation:"Max pooling takes max activation per window. Provides translation invariance." },
      { q:"LSTM solves vanishing gradient using:", options:["Deeper networks","Gating mechanisms with additive cell state","Larger learning rate","Weight initialisation"], answer:1, explanation:"LSTM cell state updated additively with gates (forget, input, output) controlling memory." }
    ],
    medium: [
      { q:"Xavier/Glorot initialisation sets weight variance to:", options:["1","2/(nᵢₙ + nₒᵤₜ)","1/nᵢₙ","Random uniform"], answer:1, explanation:"Xavier: Var(w) = 2/(nᵢₙ+nₒᵤₜ). He initialisation: 2/nᵢₙ for ReLU." },
      { q:"Residual connections (ResNets) help by:", options:["Reducing parameters","Providing gradient highways: y = F(x)+x","Increasing non-linearity","Removing need for BN"], answer:1, explanation:"Skip connections let gradient flow through identity shortcut → trains very deep nets." },
      { q:"Transformer self-attention computes:", options:["Fixed weights","Attention(Q,K,V) = softmax(QKᵀ/√d)V","Convolution on sequence","Recurrent states"], answer:1, explanation:"Self-attention: scaled dot-product attention allows each position to attend to all others." },
      { q:"Knowledge distillation trains small model by:", options:["Adding more parameters","Matching soft outputs (logits) of large teacher","Using more data","Increasing learning rate"], answer:1, explanation:"Distillation: train student to match teacher's soft logits. Soft targets carry more info than hard labels." },
      { q:"L1 vs L2 weight regularisation:", options:["L1 prevents overfitting, L2 doesn't","L1 promotes sparsity; L2 promotes small uniform weights","L2 promotes sparsity","Both identical"], answer:1, explanation:"L1: weights snap to 0 (sparse). L2/weight decay: weights shrink uniformly but rarely reach 0." },
      { q:"Receptive field of a CNN layer refers to:", options:["Number of filters","Input region affecting one output neuron","Stride value","Padding amount"], answer:1, explanation:"Receptive field grows with depth. RF = (k-1)×L + 1 for L layers of kernel k." },
      { q:"Gradient clipping prevents:", options:["Vanishing gradients","Exploding gradients","Slow convergence","Overfitting"], answer:1, explanation:"Clip when ||g|| > threshold: scale g → (threshold/||g||)×g. Prevents parameter explosion in RNNs." },
      { q:"Positional encoding in Transformers adds:", options:["Word embeddings","Sequential position information (attention is permutation-invariant)","Attention weights","Layer normalisation"], answer:1, explanation:"PE adds sin/cos functions of position to embeddings so model knows token order." },
      { q:"Depthwise separable convolution vs standard:", options:["Same computation","Separates spatial and channel ops — much cheaper","More parameters","Lower accuracy always"], answer:1, explanation:"Separates 2D per-channel filter + 1×1 pointwise. Reduces computation ~k× per layer." },
      { q:"Encoder-decoder architecture is used for:", options:["Classification only","Sequence-to-sequence tasks (translation, summarisation)","Embedding only","Image classification only"], answer:1, explanation:"Encoder compresses input; decoder generates output. Used in MT, summarisation, segmentation." },
      { q:"Mode collapse in GANs means:", options:["Discriminator always wins","Generator produces limited variety, ignoring data diversity","Training diverges","Gradient vanishes"], answer:1, explanation:"Mode collapse: generator finds few 'safe' modes. Solutions: WGAN, minibatch discrimination." },
      { q:"Self-attention complexity with sequence length n is:", options:["O(n)","O(n²)","O(n log n)","O(1)"], answer:1, explanation:"Self-attention: n×n attention matrix → O(n²d). Bottleneck for long sequences." },
      { q:"Label smoothing replaces hard labels with:", options:["Random labels","Soft labels: ε/(K-1) and 1-ε","Logit values","Gradient values"], answer:1, explanation:"Label smoothing: y_smooth = (1-ε)y + ε/K. Prevents overconfidence, improves calibration." },
      { q:"Neural Architecture Search (NAS) automates:", options:["Weight training","Designing the neural network architecture","Data augmentation","Loss function design"], answer:1, explanation:"NAS uses RL, evolutionary algorithms, or differentiable search to find optimal architecture." }
    ],
    hard: [
      { q:"VAE loss consists of:", options:["Reconstruction loss only","Reconstruction loss + KL divergence between encoder and prior","MSE only","Cross-entropy only"], answer:1, explanation:"VAE: L = -E[log p(x|z)] + KL(q(z|x)||p(z)). Reconstruction + latent space regularisation." },
      { q:"Reparameterisation trick in VAE allows:", options:["Faster inference","Backprop through stochastic sampling: z = μ + σ·ε","Higher-dim latent space","Discrete latent variables"], answer:1, explanation:"z = μ + σ·ε, ε~N(0,I). Moves randomness to ε → gradient ∂z/∂(μ,σ) is tractable." },
      { q:"WGAN addresses GAN instability by:", options:["Removing discriminator","Providing smoother gradient signal when distributions don't overlap","Binary cross-entropy","Clipping generator weights"], answer:1, explanation:"Wasserstein distance defined even for non-overlapping distributions → meaningful gradient always." },
      { q:"Multi-head attention uses h heads to:", options:["Reduce computation","Attend from different representation subspaces simultaneously","Replace all layers","Increase sequence length"], answer:1, explanation:"Multi-head: h parallel attention functions on projected Q,K,V (d/h each). Captures diverse relationships." },
      { q:"Gradient checkpointing trades:", options:["Accuracy for speed","Computation for memory by recomputing activations during backprop","Memory for accuracy","Speed for regularisation"], answer:1, explanation:"Checkpointing: recompute activations in backward pass. Memory: O(√layers) instead of O(layers)." },
      { q:"Normalising flows learn:", options:["Fixed distributions","Invertible transforms with exact likelihood via change of variables","Approximate posteriors","Discriminative classifiers"], answer:1, explanation:"NF: log p(x) = log p(z) + Σlog|det Jᵢ| using invertible transformations. Exact likelihood." },
      { q:"Lottery Ticket Hypothesis states:", options:["All networks equivalent","Dense networks contain sparse subnetworks that match full network accuracy","Large models always overfit","Random pruning works best"], answer:1, explanation:"Within random init, a sparse subnetwork ('winning ticket') can be trained to match full network." },
      { q:"Mamba (state space models) improves on Transformers by:", options:["More attention heads","Selective state space with O(n) vs O(n²) complexity","Larger embeddings","Removing positional encoding"], answer:1, explanation:"Mamba: input-dependent selection in SSM → O(n) sequence complexity vs O(n²) attention." },
      { q:"Neural Tangent Kernel (NTK) regime: width → ∞ means:", options:["Deeper networks","Training dynamics become linear — network behaves as kernel method","Dropout rate is 1","Batch size is 1"], answer:1, explanation:"Infinite-width limit: NTK stays constant during training → equivalent to kernel regression." },
      { q:"Mixture of Experts (MoE) scales LLMs by:", options:["Deeper networks","Routing each token to few experts: more params, same compute","More attention heads","Reducing vocabulary"], answer:1, explanation:"Sparse MoE: top-k expert routing keeps compute constant while scaling parameters. Used in GPT-4." },
      { q:"Flash Attention optimises by:", options:["Reducing parameters","Tiling attention in GPU SRAM to avoid slow HBM reads","Approximate attention","Reducing sequence length"], answer:1, explanation:"Flash Attention: IO-aware exact attention. Avoids materialising full n×n matrix in slow HBM. 2-4× speedup." },
      { q:"Constitutional AI (RLAIF) trains models to:", options:["Use only human feedback","Follow principles using AI feedback on its own outputs without human labelling","Generate constitutions","Avoid all refusals"], answer:1, explanation:"Constitutional AI: model critiques/revises its own outputs per a constitution. Reduces harmful outputs without extensive human annotation." }
    ]
  }
};

// ── QUIZ ENGINE ────────────────────────────────
let currentQuiz    = [];
let currentIndex   = 0;
let score          = 0;
let answered       = false;
let startTime      = null;

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function buildQuiz(subject, level) {
  const pool = questionBank[subject][level];
  currentQuiz  = shuffle([...pool]).slice(0, Math.min(10, pool.length));
  currentIndex = 0;
  score        = 0;
  answered     = false;
  startTime    = Date.now();
  renderQuestion();
}

function renderQuestion() {
  const area = document.getElementById('quizArea');
  const rs   = document.getElementById('resultScreen');
  if (!area) return;
  if (rs) rs.innerHTML = '';

  if (currentIndex >= currentQuiz.length) { showResult(); return; }

  const q   = currentQuiz[currentIndex];
  const num = currentIndex + 1;
  const tot = currentQuiz.length;

  area.innerHTML = `
    <div class="quiz-card">
      <div class="quiz-progress">
        <span>Question ${num} of ${tot}</span>
        <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${(num/tot)*100}%"></div></div>
        <span>Score: ${score}</span>
      </div>
      <p class="quiz-question">${q.q}</p>
      <div class="quiz-options" id="options">
        ${q.options.map((opt, i) => `
          <button class="option-btn" id="opt${i}" onclick="selectAnswer(${i})">
            <span class="opt-letter">${String.fromCharCode(65+i)}</span> ${opt}
          </button>`).join('')}
      </div>
      <div id="explanation" class="explanation hidden"></div>
      <div id="nextWrap" class="hidden" style="text-align:right;margin-top:1rem">
        <button class="btn btn-primary" onclick="nextQuestion()">
          ${currentIndex < currentQuiz.length - 1 ? 'Next Question →' : 'See Results 🎉'}
        </button>
      </div>
    </div>`;
}

function selectAnswer(chosen) {
  if (answered) return;
  answered = true;
  const q    = currentQuiz[currentIndex];
  const btns = document.querySelectorAll('.option-btn');

  btns.forEach((btn, i) => {
    btn.disabled = true;
    if (i === q.answer) btn.classList.add('correct');
    else if (i === chosen) btn.classList.add('wrong');
  });

  if (chosen === q.answer) score++;

  const exp = document.getElementById('explanation');
  if (exp) {
    exp.innerHTML = `<strong>${chosen === q.answer ? '✅ Correct!' : '❌ Incorrect.'}</strong><br>${q.explanation}`;
    exp.classList.remove('hidden');
  }
  const nw = document.getElementById('nextWrap');
  if (nw) nw.classList.remove('hidden');
}

function nextQuestion() {
  currentIndex++;
  answered = false;
  renderQuestion();
}

function showResult() {
  const area = document.getElementById('quizArea');
  const rs   = document.getElementById('resultScreen');
  if (area) area.innerHTML = '';
  if (!rs)  return;

  const pct      = Math.round((score / currentQuiz.length) * 100);
  const timeSecs = Math.round((Date.now() - startTime) / 1000);
  const fb       = getFeedback(pct);

  rs.innerHTML = `
    <div class="quiz-card result-card">
      <div style="font-size:3.5rem;margin-bottom:.5rem">${fb.emoji}</div>
      <h2 style="font-family:'DM Serif Display',serif;color:var(--text-dark)">${fb.message}</h2>
      <div class="result-score">${score} / ${currentQuiz.length}</div>
      <div class="result-pct">${pct}%</div>
      <p style="color:var(--text-mid);margin:1rem 0">${fb.tip}</p>
      <p style="color:var(--text-light);font-size:.85rem">⏱ Completed in ${timeSecs}s</p>
      <div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-top:1.5rem">
        <button class="btn btn-primary" onclick="startQuiz('${fb.nextLevel}')">Try ${fb.nextLevel} mode 🔄</button>
        <button class="btn btn-secondary" onclick="initQuizSelector()">Change Subject 📚</button>
      </div>
    </div>`;
  rs.scrollIntoView({ behavior:'smooth', block:'start' });
}

function getFeedback(pct) {
  if (pct >= 90) return { emoji:'🏆', message:"Outstanding! You're exam-ready!", tip:`${pct}% is excellent! Keep practising Hard level to stay sharp. You're going to ace EC3! 🌟`, nextLevel:'hard' };
  if (pct >= 75) return { emoji:'⭐', message:"Great work! Almost there!", tip:`${pct}% is very solid. Focus on the questions you got wrong. Keep it up! 💪`, nextLevel:'hard' };
  if (pct >= 65) return { emoji:'💪', message:"Improving steadily — keep going!", tip:`${pct}% is solid progress. Review explanations for wrong answers — fastest improvement area. ✨`, nextLevel:'medium' };
  if (pct >= 40) return { emoji:'🌱', message:"Good start! Build on it.", tip:`${pct}% means you have the fundamentals. Revisit notes then retry. Small steps = big success! 📚`, nextLevel:'easy' };
  return { emoji:'💙', message:"Every expert was once a beginner.", tip:`${pct}% tells you what to focus on. Start Easy to build confidence, then climb up. Read explanations carefully! 🔄`, nextLevel:'easy' };
}

// ── QUIZ SELECTOR ───────────────────────────
let selectedSubject = 'MFML';

function initQuizSelector() {
  const selector = document.getElementById('quizSelector');
  if (!selector) return;

  const subjects = [
    { sub:'MFML', icon:'∑', label:'Mathematics for ML',    diff:'Toughest',     total: Object.values(questionBank.MFML).flat().length },
    { sub:'ISM',  icon:'📊', label:'Statistical Methods',  diff:'Most Scoring', total: Object.values(questionBank.ISM).flat().length  },
    { sub:'ML',   icon:'🤖', label:'Machine Learning',     diff:'Moderate',     total: Object.values(questionBank.ML).flat().length   },
    { sub:'DNN',  icon:'🧠', label:'Deep Neural Networks', diff:'Hard',         total: Object.values(questionBank.DNN).flat().length  }
  ];

  selector.innerHTML = `
    <div style="text-align:center;margin-bottom:2rem">
      <div class="hero-badge">🎯 Adaptive Mock Test Engine</div>
      <h2 style="font-family:'DM Serif Display',serif;color:var(--text-dark);margin-top:1rem">Choose Your Practice Session</h2>
      <p style="color:var(--text-mid);margin-top:.5rem">Select subject → difficulty → 10 random questions drawn from large bank</p>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:1.5rem;margin-bottom:2rem">
      ${subjects.map(s=>`
        <div class="subject-card" id="card-${s.sub}" onclick="selectSubject('${s.sub}')">
          <div style="font-size:2rem;margin-bottom:.5rem">${s.icon}</div>
          <div class="card-badge">${s.diff}</div>
          <h3 style="font-weight:800;color:var(--text-dark);margin:.4rem 0">${s.sub}</h3>
          <p style="font-size:.8rem;color:var(--text-mid);margin-bottom:.25rem">${s.label}</p>
          <p style="font-size:.75rem;color:var(--accent-purple);font-weight:700">${s.total} questions available</p>
        </div>`).join('')}
    </div>
    <div id="levelPicker" style="display:none;background:var(--lavender-soft);border-radius:16px;padding:1.5rem;margin-bottom:2rem">
      <h3 style="font-family:'DM Serif Display',serif;color:var(--text-dark);margin-bottom:1rem">Select Difficulty</h3>
      <div style="display:flex;gap:.75rem;flex-wrap:wrap">
        <button class="btn btn-secondary" onclick="startQuiz('easy')">🌱 Easy</button>
        <button class="btn btn-secondary" onclick="startQuiz('medium')">⚡ Medium</button>
        <button class="btn btn-secondary" onclick="startQuiz('hard')">🔥 Hard</button>
      </div>
      <p style="font-size:.8rem;color:var(--text-light);margin-top:.75rem">10 questions per session · Randomised · Explanations shown after each answer</p>
    </div>
    <div id="quizArea"></div>
    <div class="result-screen" id="resultScreen"></div>`;
}

function selectSubject(sub) {
  selectedSubject = sub;
  document.getElementById('levelPicker').style.display = 'block';
  document.getElementById('levelPicker').scrollIntoView({ behavior:'smooth', block:'center' });
  document.querySelectorAll('.subject-card').forEach(c => c.style.outline = '');
  const card = document.getElementById('card-'+sub);
  if (card) card.style.outline = '3px solid var(--accent-purple)';
}

function startQuiz(level) {
  buildQuiz(selectedSubject, level);
  document.getElementById('quizArea').scrollIntoView({ behavior:'smooth', block:'start' });
}

document.addEventListener('DOMContentLoaded', initQuizSelector);
