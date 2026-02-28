// =============================================
//  BITS WILP EC3 Prep Hub — Mock Test Engine
//  Adaptive Logic + Motivational Results
// =============================================

// ── QUESTION BANK ─────────────────────────────
const questionBank = {

  // ===== MFML QUESTIONS =====
  MFML: {
    easy: [
      {
        q: "Which of the following is a property of a symmetric matrix?",
        options: ["A = Aᵀ", "A = -Aᵀ", "A has no eigenvalues", "A is always invertible"],
        answer: 0,
        explanation: "A symmetric matrix satisfies A = Aᵀ (equals its own transpose). All real symmetric matrices are guaranteed to have real eigenvalues."
      },
      {
        q: "The dot product of two orthogonal vectors is:",
        options: ["1", "-1", "0", "Undefined"],
        answer: 2,
        explanation: "Orthogonal vectors are perpendicular to each other. Their dot product is 0, i.e., u·v = |u||v|cos(90°) = 0."
      },
      {
        q: "What is the rank of a zero matrix?",
        options: ["0", "1", "n (number of rows)", "Undefined"],
        answer: 0,
        explanation: "A zero matrix has all entries equal to zero. The rank equals the number of linearly independent rows/columns, which is 0 for a zero matrix."
      },
      {
        q: "If eigenvalue of matrix A is λ, what is the eigenvalue of A²?",
        options: ["2λ", "λ²", "√λ", "1/λ"],
        answer: 1,
        explanation: "If Av = λv, then A²v = A(Av) = A(λv) = λ(Av) = λ²v. So the eigenvalue of A² is λ²."
      }
    ],
    medium: [
      {
        q: "The gradient of f(x,y) = x²y + y³ at point (1, 2) is:",
        options: ["[4, 13]", "[4, 12]", "[2, 13]", "[4, 11]"],
        answer: 0,
        explanation: "∂f/∂x = 2xy = 2(1)(2) = 4. ∂f/∂y = x² + 3y² = 1 + 12 = 13. So ∇f(1,2) = [4, 13]."
      },
      {
        q: "A matrix A is positive definite if and only if:",
        options: [
          "All eigenvalues are positive",
          "All diagonal entries are positive",
          "Determinant is positive",
          "All entries are positive"
        ],
        answer: 0,
        explanation: "A symmetric matrix is positive definite iff all its eigenvalues are strictly positive. This ensures xᵀAx > 0 for all non-zero vectors x."
      },
      {
        q: "The Frobenius norm of a matrix A with entries aᵢⱼ is defined as:",
        options: [
          "Σᵢⱼ aᵢⱼ",
          "√(Σᵢⱼ aᵢⱼ²)",
          "max|aᵢⱼ|",
          "Σᵢ|aᵢᵢ|"
        ],
        answer: 1,
        explanation: "The Frobenius norm is the square root of the sum of squares of all entries: ||A||_F = √(Σᵢⱼ aᵢⱼ²). It generalizes the vector L2 norm to matrices."
      },
      {
        q: "Singular Value Decomposition (SVD) decomposes matrix A into:",
        options: ["L·D·Lᵀ", "Q·R", "U·Σ·Vᵀ", "P·D·P⁻¹"],
        answer: 2,
        explanation: "SVD decomposes A = UΣVᵀ where U, V are orthogonal matrices and Σ is diagonal with singular values. This works for any m×n matrix."
      }
    ],
    hard: [
      {
        q: "If A is an m×n matrix with m < n, what can we say about its null space?",
        options: [
          "Null space is trivial (only zero vector)",
          "Null space has dimension ≥ n - m",
          "Null space has dimension exactly m",
          "Null space is undefined for non-square matrices"
        ],
        answer: 1,
        explanation: "By the Rank-Nullity theorem: rank(A) + nullity(A) = n. Since rank(A) ≤ m < n, we get nullity(A) = n - rank(A) ≥ n - m. Non-trivial null space exists."
      },
      {
        q: "The KKT conditions in constrained optimization include:",
        options: [
          "Stationarity, primal feasibility, dual feasibility, complementary slackness",
          "Only gradient equals zero",
          "Hessian must be positive definite",
          "Objective must be convex"
        ],
        answer: 0,
        explanation: "KKT conditions are necessary conditions for a solution of a non-linear program: (1) Stationarity: ∇L=0, (2) Primal feasibility, (3) Dual feasibility: λᵢ≥0, (4) Complementary slackness: λᵢgᵢ=0."
      },
      {
        q: "For gradient descent with learning rate α to converge, which condition on the loss function L is sufficient?",
        options: [
          "L is bounded below and its gradient is Lipschitz continuous",
          "L is convex",
          "L has a unique minimum",
          "L is twice differentiable"
        ],
        answer: 0,
        explanation: "If L is bounded below and ∇L is L-Lipschitz (||∇L(x)-∇L(y)||≤L||x-y||), then gradient descent with α ≤ 1/L converges to a critical point. Convexity alone doesn't guarantee convergence rate."
      }
    ]
  },

  // ===== ISM QUESTIONS =====
  ISM: {
    easy: [
      {
        q: "The expected value of a Bernoulli(p) random variable is:",
        options: ["p(1-p)", "p²", "p", "1-p"],
        answer: 2,
        explanation: "A Bernoulli(p) variable X takes value 1 with probability p and 0 with probability 1-p. E[X] = 1·p + 0·(1-p) = p."
      },
      {
        q: "If X ~ N(μ, σ²), what is the distribution of (X - μ)/σ?",
        options: ["N(μ, σ²)", "N(0, σ²)", "N(0, 1)", "N(1, 1)"],
        answer: 2,
        explanation: "Standardizing X by subtracting mean μ and dividing by standard deviation σ gives Z = (X-μ)/σ ~ N(0,1), the standard normal distribution."
      },
      {
        q: "The variance of a constant c is:",
        options: ["c", "c²", "1", "0"],
        answer: 3,
        explanation: "A constant never varies. Var(c) = E[(c - E[c])²] = E[(c-c)²] = 0. Constants have zero variance."
      },
      {
        q: "In a hypothesis test, Type I error means:",
        options: [
          "Rejecting H₀ when it is true",
          "Failing to reject H₀ when it is false",
          "Accepting H₁ when H₀ is true",
          "Both A and C"
        ],
        answer: 3,
        explanation: "Type I error (α) is rejecting a true null hypothesis — a 'false positive'. Since rejecting H₀ is equivalent to accepting H₁, both A and C describe the same event."
      }
    ],
    medium: [
      {
        q: "For a Poisson process with rate λ, the variance of inter-arrival times is:",
        options: ["λ", "1/λ", "1/λ²", "λ²"],
        answer: 2,
        explanation: "Inter-arrival times of a Poisson(λ) process follow Exponential(λ) distribution. For Exp(λ): Mean = 1/λ, Variance = 1/λ²."
      },
      {
        q: "The Central Limit Theorem states that for large n, the sample mean X̄ is approximately:",
        options: [
          "N(μ, σ²)",
          "N(μ, σ²/n)",
          "N(0, 1)",
          "N(μ, σ/n)"
        ],
        answer: 1,
        explanation: "CLT: If X₁,...,Xₙ are i.i.d. with mean μ and variance σ², then X̄ ~ N(μ, σ²/n) approximately for large n. Standard error of mean = σ/√n."
      },
      {
        q: "A 95% confidence interval for population mean μ when σ is known is:",
        options: [
          "X̄ ± 1.645(σ/√n)",
          "X̄ ± 1.96(σ/√n)",
          "X̄ ± 2.576(σ/√n)",
          "X̄ ± 1.96(σ)"
        ],
        answer: 1,
        explanation: "For 95% CI with known σ, z* = 1.96 (z₀.₀₂₅). CI = X̄ ± 1.96(σ/√n). Remember: 90%→1.645, 95%→1.96, 99%→2.576."
      },
      {
        q: "Bayes' theorem states P(A|B) equals:",
        options: [
          "P(B|A)P(A)/P(B)",
          "P(A)P(B)/P(A∩B)",
          "P(A∩B)/P(A)",
          "P(B|A)/P(B)"
        ],
        answer: 0,
        explanation: "Bayes' theorem: P(A|B) = P(B|A)P(A)/P(B). P(B|A) is likelihood, P(A) is prior, P(B) is evidence (normalizing constant), P(A|B) is posterior."
      }
    ],
    hard: [
      {
        q: "The MLE estimate for the parameter λ of an Exponential distribution given observations x₁,...,xₙ is:",
        options: ["n/Σxᵢ", "Σxᵢ/n", "√(Σxᵢ/n)", "n·Σxᵢ"],
        answer: 0,
        explanation: "Log-likelihood of Exp(λ): ℓ(λ) = n·log(λ) - λΣxᵢ. Setting dℓ/dλ = n/λ - Σxᵢ = 0 gives λ̂_MLE = n/Σxᵢ = 1/x̄."
      },
      {
        q: "In linear regression Y = Xβ + ε, the OLS estimator β̂ is BLUE (Best Linear Unbiased Estimator) when:",
        options: [
          "ε ~ N(0, σ²I) — Gauss-Markov conditions hold",
          "X is invertible",
          "Y is normally distributed",
          "Sample size n > p"
        ],
        answer: 0,
        explanation: "By Gauss-Markov theorem, OLS is BLUE when: (1) E[ε]=0, (2) Var(ε)=σ²I (homoscedasticity + no autocorrelation), (3) X is non-stochastic with full rank. Normality is NOT required for BLUE."
      },
      {
        q: "The p-value in hypothesis testing represents:",
        options: [
          "Probability that the null hypothesis is true",
          "Probability of observing data as extreme or more extreme than observed, assuming H₀ is true",
          "Probability of Type II error",
          "The significance level α"
        ],
        answer: 1,
        explanation: "The p-value = P(T ≥ t_obs | H₀). A small p-value means the observed data is unlikely under H₀, providing evidence against H₀. It does NOT give P(H₀ is true)."
      }
    ]
  },

  // ===== ML QUESTIONS =====
  ML: {
    easy: [
      {
        q: "Which algorithm is used for supervised classification of linearly separable data?",
        options: ["K-Means", "PCA", "Support Vector Machine (SVM)", "Apriori"],
        answer: 2,
        explanation: "SVM finds the optimal hyperplane that maximally separates classes with maximum margin. For linearly separable data, the hard-margin SVM finds the exact separating hyperplane."
      },
      {
        q: "Overfitting occurs when a model:",
        options: [
          "Performs well on training data but poorly on unseen data",
          "Performs poorly on both training and test data",
          "Performs well on test data but poorly on training data",
          "Has too few parameters"
        ],
        answer: 0,
        explanation: "Overfitting: model memorizes training data including noise, resulting in high training accuracy but low generalization. The model fails on new, unseen data."
      },
      {
        q: "In k-fold cross-validation, the dataset is divided into:",
        options: [
          "k training sets",
          "k equal folds, each used as validation set once",
          "k random samples with replacement",
          "k independent test sets"
        ],
        answer: 1,
        explanation: "k-fold CV: data is split into k folds. Model trains on k-1 folds and validates on the remaining 1, rotating k times. Final metric is averaged across all k runs."
      }
    ],
    medium: [
      {
        q: "The kernel trick in SVM allows:",
        options: [
          "Faster training convergence",
          "Computation of inner products in high-dimensional feature space without explicit mapping",
          "Reduction of support vectors",
          "Elimination of the regularization parameter C"
        ],
        answer: 1,
        explanation: "The kernel trick computes K(xᵢ,xⱼ) = φ(xᵢ)·φ(xⱼ) without explicitly computing φ(x). This enables SVMs to work in infinite-dimensional spaces (e.g., RBF kernel) efficiently."
      },
      {
        q: "In a Random Forest, individual decision trees are trained using:",
        options: [
          "The full training dataset for each tree",
          "Bootstrap samples (sampling with replacement) + random feature subsets",
          "Sequential residuals from previous trees",
          "Stratified splits only"
        ],
        answer: 1,
        explanation: "Random Forest combines bagging (bootstrap sampling) + random feature selection at each split. This decorrelates trees and reduces overfitting compared to a single deep tree."
      },
      {
        q: "The bias-variance tradeoff states that total expected error equals:",
        options: [
          "Bias² + Variance",
          "Bias + Variance + Noise",
          "Bias² + Variance + Irreducible Noise",
          "Bias² / Variance"
        ],
        answer: 2,
        explanation: "E[(y - ŷ)²] = Bias² + Variance + Irreducible Noise. High bias = underfitting (model too simple). High variance = overfitting (model too complex). Goal: find the sweet spot."
      },
      {
        q: "L2 regularization (Ridge) adds which penalty term to the loss function?",
        options: ["λΣ|wᵢ|", "λΣwᵢ²", "λmax|wᵢ|", "λΣ√|wᵢ|"],
        answer: 1,
        explanation: "Ridge (L2) regularization adds λ||w||₂² = λΣwᵢ² to the loss. This penalizes large weights, leading to small but non-zero weights. Unlike Lasso (L1), Ridge doesn't produce sparse solutions."
      }
    ],
    hard: [
      {
        q: "The EM algorithm alternates between:",
        options: [
          "Gradient descent and Newton's method",
          "E-step (compute expected complete-data log-likelihood) and M-step (maximize it)",
          "Feature selection and model training",
          "Initialization and convergence checking"
        ],
        answer: 1,
        explanation: "EM: E-step computes Q(θ|θᵗ) = E[log P(X,Z|θ) | X, θᵗ] over latent variables Z. M-step finds θᵗ⁺¹ = argmax Q. EM is guaranteed to increase the marginal likelihood P(X|θ) monotonically."
      },
      {
        q: "For a Gaussian Mixture Model with K components, the E-step computes:",
        options: [
          "New means of each Gaussian",
          "The responsibility rᵢₖ = P(component k | xᵢ, θ) using Bayes' theorem",
          "The optimal number of components",
          "The covariance matrices"
        ],
        answer: 1,
        explanation: "E-step computes responsibilities rᵢₖ = πₖN(xᵢ|μₖ,Σₖ) / Σⱼπⱼ N(xᵢ|μⱼ,Σⱼ). These are soft assignments of each point to each Gaussian component."
      }
    ]
  },

  // ===== DNN QUESTIONS =====
  DNN: {
    easy: [
      {
        q: "The activation function ReLU is defined as:",
        options: ["max(0, x)", "1/(1+e⁻ˣ)", "tanh(x)", "x²"],
        answer: 0,
        explanation: "ReLU (Rectified Linear Unit): f(x) = max(0, x). It's computationally efficient, avoids vanishing gradients in positive region, but can suffer from 'dying ReLU' when outputs become 0."
      },
      {
        q: "Backpropagation computes gradients using:",
        options: [
          "Forward pass only",
          "Chain rule of calculus applied in reverse through the network",
          "Numerical differentiation",
          "Random gradient estimation"
        ],
        answer: 1,
        explanation: "Backprop applies the chain rule backwards: δL/δw = δL/δa · δa/δz · δz/δw. Gradients flow backward from output to input layer, enabling efficient weight updates."
      },
      {
        q: "Dropout regularization during training:",
        options: [
          "Removes all neurons with negative activations",
          "Randomly sets a fraction of neuron outputs to zero",
          "Reduces the learning rate adaptively",
          "Normalizes activations between layers"
        ],
        answer: 1,
        explanation: "Dropout randomly zeroes neuron outputs with probability p during training. This prevents co-adaptation of neurons, forces redundant representations, and acts as an ensemble of 2^n networks."
      }
    ],
    medium: [
      {
        q: "The vanishing gradient problem in deep networks occurs because:",
        options: [
          "Learning rate is too large",
          "Gradients shrink exponentially as they propagate backward through many layers with saturating activations",
          "Batch size is too small",
          "Too many parameters exist"
        ],
        answer: 1,
        explanation: "In deep networks with sigmoid/tanh activations, gradients are multiplied by derivatives (≤0.25 for sigmoid) at each layer. Over many layers, gradients exponentially vanish: δL/δw₁ ≈ 0.25ᴸ for L layers."
      },
      {
        q: "Batch Normalization normalizes layer inputs to have approximately:",
        options: [
          "Zero minimum and unit maximum",
          "Zero mean and unit variance, then rescales with learnable γ and β",
          "Same distribution as the training data",
          "Uniform distribution"
        ],
        answer: 1,
        explanation: "BatchNorm: x̂ = (x-μ_B)/√(σ²_B+ε), then y = γx̂ + β. This reduces internal covariate shift, allows higher learning rates, acts as regularization, and reduces sensitivity to initialization."
      },
      {
        q: "In a Convolutional Neural Network, the purpose of pooling layers is to:",
        options: [
          "Increase the spatial dimensions of feature maps",
          "Add non-linearity to the network",
          "Reduce spatial dimensions, providing translation invariance and reducing parameters",
          "Normalize the gradients"
        ],
        answer: 2,
        explanation: "Pooling (max/average) downsamples feature maps: reduces spatial size, provides local translation invariance, reduces computation, and helps control overfitting by reducing parameters."
      },
      {
        q: "The attention mechanism in Transformers computes:",
        options: [
          "Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V",
          "Attention(Q,K,V) = sigmoid(QKᵀ)V",
          "Attention(Q,K,V) = QKᵀV/dₖ",
          "Attention(Q,K,V) = max(QKᵀ)V"
        ],
        answer: 0,
        explanation: "Scaled dot-product attention: Attention = softmax(QKᵀ/√dₖ)V. Division by √dₖ prevents vanishingly small gradients from softmax in high dimensions. Q=queries, K=keys, V=values."
      }
    ],
    hard: [
      {
        q: "The LSTM cell uses gates to address the vanishing gradient problem. The forget gate computes:",
        options: [
          "fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)",
          "fₜ = tanh(Wf·[hₜ₋₁, xₜ] + bf)",
          "fₜ = ReLU(Wf·[hₜ₋₁, xₜ] + bf)",
          "fₜ = Wf·[hₜ₋₁, xₜ]"
        ],
        answer: 0,
        explanation: "LSTM forget gate: fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf). The sigmoid output ∈(0,1) decides what fraction of cell state to retain. Cell state update: Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ."
      },
      {
        q: "Knowledge distillation trains a smaller student network by:",
        options: [
          "Using only hard labels from the original dataset",
          "Using soft probability outputs (soft targets) from a trained teacher network",
          "Copying weights directly from the teacher",
          "Using the teacher network's activations as regularization"
        ],
        answer: 1,
        explanation: "Knowledge distillation (Hinton et al.): student trained on soft targets P(y|x;T) from teacher (temperature T>1 softens distribution). Soft targets carry more information than one-hot labels, enabling efficient knowledge transfer."
      }
    ]
  }
};

// ── QUIZ STATE ─────────────────────────────────
let currentTest = [];
let currentQ    = 0;
let score       = 0;
let userAnswers = [];
let quizSubject = 'MFML';
let quizLevel   = 'medium';

// ── BUILD QUIZ ─────────────────────────────────
function buildQuiz(subject, level) {
  quizSubject = subject;
  quizLevel   = level;
  currentQ    = 0;
  score       = 0;
  userAnswers = [];

  const pool = questionBank[subject][level] || [];
  // Shuffle and take up to 8
  currentTest = [...pool].sort(() => Math.random() - 0.5);

  showQuestion();
  document.getElementById('resultScreen')?.classList.remove('show');
  document.getElementById('quizArea')?.style.removeProperty('display');
}

// ── SHOW QUESTION ──────────────────────────────
function showQuestion() {
  const container = document.getElementById('quizArea');
  if (!container) return;

  if (currentQ >= currentTest.length) {
    showResult();
    return;
  }

  const q = currentTest[currentQ];
  const pct = Math.round((currentQ / currentTest.length) * 100);

  container.innerHTML = `
    <div class="progress-bar-wrap">
      <div class="progress-bar-fill" style="width:${pct}%"></div>
    </div>
    <div class="question-card">
      <div class="question-num">Question ${currentQ + 1} of ${currentTest.length}</div>
      <div class="question-text">${q.q}</div>
      <div class="options-grid" id="options"></div>
      <div class="explanation-box" id="explanation">
        <strong>💡 Explanation:</strong> ${q.explanation}
      </div>
    </div>
    <div style="text-align:right; margin-top:1rem">
      <button class="btn btn-primary" id="nextBtn" onclick="nextQuestion()" style="display:none">
        ${currentQ + 1 === currentTest.length ? '📊 View Results' : 'Next Question →'}
      </button>
    </div>
  `;

  const letters = ['A','B','C','D'];
  const optionsContainer = document.getElementById('options');
  q.options.forEach((opt, i) => {
    const btn = document.createElement('button');
    btn.className = 'option-btn';
    btn.innerHTML = `<span class="option-letter">${letters[i]}</span> ${opt}`;
    btn.onclick = () => selectAnswer(i, q.answer);
    optionsContainer.appendChild(btn);
  });
}

// ── SELECT ANSWER ──────────────────────────────
function selectAnswer(selected, correct) {
  userAnswers.push({ selected, correct });

  const btns = document.querySelectorAll('.option-btn');
  btns.forEach(btn => { btn.onclick = null; btn.style.cursor = 'default'; });

  btns[selected].classList.add(selected === correct ? 'correct' : 'wrong');
  if (selected !== correct) btns[correct].classList.add('correct');
  if (selected === correct) score++;

  document.getElementById('explanation')?.classList.add('show');
  document.getElementById('nextBtn').style.display = 'inline-flex';
}

// ── NEXT QUESTION ──────────────────────────────
function nextQuestion() {
  currentQ++;
  showQuestion();
}

// ── SHOW RESULT ────────────────────────────────
function showResult() {
  document.getElementById('quizArea').style.display = 'none';
  const total = currentTest.length;
  const pct   = Math.round((score / total) * 100);

  const { emoji, message, tip, nextLevel } = getResultContent(pct);

  const screen = document.getElementById('resultScreen');
  screen.classList.add('show');
  screen.innerHTML = `
    <div class="score-circle">
      <div>
        <div class="score-num">${score}/${total}</div>
        <div class="score-denom">${pct}%</div>
      </div>
    </div>
    <div class="result-message">${emoji} ${message}</div>
    <div class="result-tip">${tip}</div>
    <p style="color:var(--text-light);font-size:0.9rem;margin-bottom:2rem">
      Subject: <strong style="color:var(--accent-purple)">${quizSubject}</strong> &nbsp;|&nbsp;
      Level: <strong style="color:var(--accent-blue)">${quizLevel.charAt(0).toUpperCase()+quizLevel.slice(1)}</strong>
    </p>
    <div style="display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
      <button class="btn btn-primary" onclick="buildQuiz('${quizSubject}','${nextLevel}')">
        🔁 Try ${nextLevel.charAt(0).toUpperCase()+nextLevel.slice(1)} Questions
      </button>
      <button class="btn btn-secondary" onclick="buildQuiz('${quizSubject}','${quizLevel}')">
        🔄 Retry Same Level
      </button>
    </div>
  `;
}

// ── RESULT CONTENT (Adaptive Messaging) ────────
function getResultContent(pct) {
  if (pct >= 85) {
    return {
      emoji: '🌟', message: "Outstanding! You've mastered this level!",
      tip: `You scored ${pct}%! You're ready for harder questions. The advanced level will push you further — you're clearly capable. Exam day will be great! 🎯`,
      nextLevel: 'hard'
    };
  } else if (pct >= 65) {
    return {
      emoji: '💪', message: "You're improving steadily!",
      tip: `${pct}% is solid progress! A few more practice rounds and you'll be well above the pass threshold. Focus on questions you got wrong — those are your fast-improvement areas. You've got this! ✨`,
      nextLevel: 'medium'
    };
  } else if (pct >= 40) {
    return {
      emoji: '🌱', message: "Great start! Keep building!",
      tip: `${pct}% shows you understand the basics. Review the explanations above carefully and revisit Unit notes. Revision + a second attempt usually shows a 20-30% improvement. Small steps daily = Big success! 📚`,
      nextLevel: 'easy'
    };
  } else {
    return {
      emoji: '💙', message: "Every expert was once a beginner.",
      tip: `Don't be discouraged — ${pct}% means you've identified exactly what to work on. Start with our Quick Revision page, then revisit Easy-level questions. The foundation questions will build your confidence fast! Practice makes perfect! 🔄`,
      nextLevel: 'easy'
    };
  }
}

// ── SELECTOR UI ────────────────────────────────
function initQuizSelector() {
  const selector = document.getElementById('quizSelector');
  if (!selector) return;

  selector.innerHTML = `
    <div class="quiz-header">
      <div class="hero-badge">🎓 Adaptive Mock Test Engine</div>
      <h2>Choose Your Practice Session</h2>
      <p style="color:var(--text-mid);margin-top:0.5rem">Select a subject and difficulty. Questions adapt based on your score.</p>
    </div>

    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:1.5rem;margin-bottom:2rem">
      ${[
        {sub:'MFML', icon:'∑', color:'purple', label:'Mathematics for ML', diff:'Toughest'},
        {sub:'ISM',  icon:'📊', color:'green',  label:'Statistical Methods', diff:'Most Scoring'},
        {sub:'ML',   icon:'🤖', color:'blue',   label:'Machine Learning', diff:'Moderate'},
        {sub:'DNN',  icon:'🧠', color:'orange', label:'Deep Neural Networks', diff:'Hard'}
      ].map(s => `
        <div class="card ${s.color}" style="cursor:pointer" onclick="selectSubject('${s.sub}')">
          <div class="card-icon" style="font-size:1.5rem">${s.icon}</div>
          <div class="card-badge">${s.diff}</div>
          <h3>${s.sub}</h3>
          <p style="font-size:0.8rem">${s.label}</p>
        </div>
      `).join('')}
    </div>

    <div id="levelPicker" style="display:none;margin-bottom:2rem">
      <h3 style="font-family:'DM Serif Display',serif;margin-bottom:1rem;color:var(--text-dark)">
        Select Difficulty Level
      </h3>
      <div style="display:flex;gap:0.75rem;flex-wrap:wrap">
        <button class="btn btn-secondary" onclick="startQuiz('easy')">
          🌱 Easy — Build Confidence
        </button>
        <button class="btn btn-secondary" onclick="startQuiz('medium')">
          ⚡ Medium — Exam Ready
        </button>
        <button class="btn btn-secondary" onclick="startQuiz('hard')">
          🔥 Hard — Challenge Mode
        </button>
      </div>
    </div>

    <div id="quizArea"></div>
    <div class="result-screen" id="resultScreen"></div>
  `;
}

let selectedSubject = 'MFML';

function selectSubject(sub) {
  selectedSubject = sub;
  document.getElementById('levelPicker').style.display = 'block';
  document.getElementById('levelPicker').scrollIntoView({ behavior: 'smooth', block: 'center' });
  // highlight selected card
  document.querySelectorAll('.card').forEach(c => c.style.outline = '');
  document.querySelectorAll('.card').forEach(c => {
    if (c.querySelector('h3')?.textContent === sub) {
      c.style.outline = '3px solid var(--accent-purple)';
    }
  });
}

function startQuiz(level) {
  buildQuiz(selectedSubject, level);
  document.getElementById('quizArea').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', initQuizSelector);
