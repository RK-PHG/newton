import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


TARGET_MODEL = "openai-community/gpt2-medium"
DRAFT_MODEL = "distilbert/distilgpt2"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# 1. 加载两个模型
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)

target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL
).to(device).eval()

draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_MODEL
).to(device).eval()


# ---------------------------------------------------------
# 一些基础函数
# ---------------------------------------------------------

@torch.no_grad()
def get_next_token_distribution(model, input_ids):
    """
    输入:
        input_ids: [1, seq_len]

    返回:
        下一个 token 的概率分布
        shape: [vocab_size]
    """

    outputs = model(input_ids=input_ids)

    # Causal LM 最后一个位置的 logits
    # 表示 P(x_{t+1} | x_1 ... x_t)
    logits = outputs.logits[0, -1, :]

    probs = F.softmax(logits, dim=-1)

    return probs


def sample_from_distribution(probs):
    """
    probs: [vocab_size]

    按照给定概率分布采样一个 token
    """

    return torch.multinomial(probs, num_samples=1).item()


# ---------------------------------------------------------
# 2. Draft Model 自回归生成 gamma 个 token
# ---------------------------------------------------------

@torch.no_grad()
def draft_generate(draft_model, prefix, gamma):
    """
    返回:

    draft_tokens:
        [x1, x2, ..., x_gamma]

    q_distributions:
        每个位置对应的 q_i(x)
    """

    current = prefix.clone()

    draft_tokens = []
    q_distributions = []

    for _ in range(gamma):

        # q_i(x)
        q = get_next_token_distribution(
            draft_model,
            current
        )

        # x_i ~ q_i(x)
        token = sample_from_distribution(q)

        draft_tokens.append(token)
        q_distributions.append(q)

        token_tensor = torch.tensor(
            [[token]],
            device=current.device
        )

        current = torch.cat(
            [current, token_tensor],
            dim=1
        )

    return draft_tokens, q_distributions


# ---------------------------------------------------------
# 3. Target Model 一次 Forward 验证 gamma 个 Draft Token
# ---------------------------------------------------------

@torch.no_grad()
def target_verify(
    target_model,
    prefix,
    draft_tokens,
):
    """
    Target 一次 forward 得到:

        p_1(x)
        p_2(x)
        ...
        p_gamma(x)

    以及全部接受后的:

        p_{gamma+1}(x)
    """

    prefix_len = prefix.shape[1]

    draft_tensor = torch.tensor(
        [draft_tokens],
        device=prefix.device
    )

    full_sequence = torch.cat(
        [prefix, draft_tensor],
        dim=1
    )

    # 关键：
    # 整个 prefix + gamma 个 token 一次送进 target
    outputs = target_model(
        input_ids=full_sequence
    )

    logits = outputs.logits[0]

    p_distributions = []

    # -----------------------------------------------------
    # GPT Causal LM:
    #
    # logits[i] 用于预测 token[i + 1]
    #
    # prefix:
    #
    #   a b c
    #
    # draft:
    #
    #   x1 x2 x3
    #
    # logits:
    #
    #   ... c位置 -> p(x1)
    #       x1位置 -> p(x2)
    #       x2位置 -> p(x3)
    #       x3位置 -> p(next)
    #
    # -----------------------------------------------------

    for i in range(len(draft_tokens)):

        logit_position = prefix_len - 1 + i

        p = F.softmax(
            logits[logit_position],
            dim=-1
        )

        p_distributions.append(p)

    # 所有 draft token 都被接受时，
    # 用这个分布再生成一个额外 token
    bonus_logits_position = (
        prefix_len
        + len(draft_tokens)
        - 1
    )

    bonus_distribution = F.softmax(
        logits[bonus_logits_position],
        dim=-1
    )

    return p_distributions, bonus_distribution


# ---------------------------------------------------------
# 4. 单轮 Speculative Sampling
# ---------------------------------------------------------

@torch.no_grad()
def speculative_round(
    prefix,
    draft_model,
    target_model,
    gamma=4,
    verbose=True,
):
    """
    执行一轮经典 speculative sampling
    """

    # =====================================================
    # Step 1:
    # Draft Model 生成 gamma 个 token
    # =====================================================

    draft_tokens, q_distributions = draft_generate(
        draft_model,
        prefix,
        gamma
    )

    if verbose:
        print("\nDraft generated:")

        for i, token in enumerate(draft_tokens):
            print(
                i,
                repr(tokenizer.decode([token]))
            )

    # =====================================================
    # Step 2:
    # Target 一次 forward
    # =====================================================

    p_distributions, bonus_distribution = target_verify(
        target_model,
        prefix,
        draft_tokens
    )

    accepted_tokens = []

    # =====================================================
    # Step 3:
    # 逐个执行接受 / 拒绝
    # =====================================================

    for i, token in enumerate(draft_tokens):

        q = q_distributions[i]
        p = p_distributions[i]

        # Draft 真实采到的 token
        q_x = q[token].item()
        p_x = p[token].item()

        # alpha(x) = min(1, p(x) / q(x))
        alpha = min(
            1.0,
            p_x / q_x
        )

        # u ~ Uniform(0, 1)
        u = torch.rand(1).item()

        if verbose:
            print(
                f"\nposition {i}:",
                repr(tokenizer.decode([token]))
            )

            print(
                f"q(x) = {q_x:.6f}"
            )

            print(
                f"p(x) = {p_x:.6f}"
            )

            print(
                f"alpha = {alpha:.6f}"
            )

            print(
                f"random u = {u:.6f}"
            )

        # -------------------------------------------------
        # 接受
        # -------------------------------------------------

        if u <= alpha:

            if verbose:
                print("ACCEPT")

            accepted_tokens.append(token)

        # -------------------------------------------------
        # 拒绝
        # -------------------------------------------------

        else:

            if verbose:
                print("REJECT")

            # residual:
            #
            # p'(x) ∝ max(0, p(x) - q(x))

            residual = torch.clamp(
                p - q,
                min=0
            )

            residual_sum = residual.sum()

            # 理论上发生 rejection 时 residual_sum > 0
            # 这里加一个数值保护
            if residual_sum <= 1e-12:

                residual = p

            else:

                residual = (
                    residual
                    / residual_sum
                )

            # 从残差分布重新采样
            resampled_token = (
                sample_from_distribution(
                    residual
                )
            )

            accepted_tokens.append(
                resampled_token
            )

            if verbose:
                print(
                    "resample:",
                    repr(
                        tokenizer.decode(
                            [resampled_token]
                        )
                    )
                )

            # 一旦某位置拒绝
            # 后面的 draft token 全部作废
            return accepted_tokens

    # =====================================================
    # 所有 gamma 个 token 都接受
    # =====================================================

    bonus_token = sample_from_distribution(
        bonus_distribution
    )

    accepted_tokens.append(
        bonus_token
    )

    if verbose:
        print("\nALL ACCEPTED")

        print(
            "bonus token:",
            repr(
                tokenizer.decode(
                    [bonus_token]
                )
            )
        )

    return accepted_tokens


# ---------------------------------------------------------
# 5. 完整 Speculative Decode
# ---------------------------------------------------------

@torch.no_grad()
def speculative_decode(
    prompt,
    max_new_tokens=40,
    gamma=4,
):

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(device)

    generated = input_ids.clone()

    num_new_tokens = 0

    while num_new_tokens < max_new_tokens:

        remaining = (
            max_new_tokens
            - num_new_tokens
        )

        # 防止最后一轮超过 max_new_tokens
        current_gamma = min(
            gamma,
            remaining
        )

        new_tokens = speculative_round(
            prefix=generated,
            draft_model=draft_model,
            target_model=target_model,
            gamma=current_gamma,
            verbose=True,
        )

        # 如果全接受，有可能返回 gamma + 1 个
        new_tokens = new_tokens[:remaining]

        new_tensor = torch.tensor(
            [new_tokens],
            device=device
        )

        generated = torch.cat(
            [generated, new_tensor],
            dim=1
        )

        num_new_tokens += len(
            new_tokens
        )

        # EOS
        if tokenizer.eos_token_id in new_tokens:
            break

    return tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )


# ---------------------------------------------------------
# 测试
# ---------------------------------------------------------

prompt = "Artificial intelligence will"

result = speculative_decode(
    prompt,
    max_new_tokens=30,
    gamma=4,
)

print("\n\n====================")
print("Final result:")
print("====================")
print(result)