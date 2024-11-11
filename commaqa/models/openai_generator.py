from openai import OpenAI

class OpenAIGenerator:
    def __init__(
        self,
        base_url,
        api_key="sk-dummy",
        engine="gpt-4o-mini",
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n", "A:", "Q:", "Wikipedia Title:"],
        retry_after_n_seconds=None,
        n=1,
        best_of=1,
        logprobs=0,
        is_instruct=True,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.engine = engine
        self.logprobs = logprobs
        self.n = n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.temperature = temperature
        self.retry_after_n_seconds = retry_after_n_seconds
        self.is_instruct = is_instruct


    def parse_prompt_to_messages(self, prompt):
        prompt = prompt.rstrip()
        segments = prompt.split('\n\n\n')

        messages = []
        for segment in segments:
            input_text, output_text = segment.split('\nA:')
            messages.extend([
                {'role': 'user', 'content': input_text.strip()},
                {'role': 'assistant', 'content': output_text.strip()}
            ])
        assert len(messages[-1]['content']) == 0
        return messages[:-1]

    def generate_text_sequence(self, prompt):
        # if self.is_instruct:
        #     messages = self.parse_prompt_to_messages(prompt)
        # else:
        #     messages = [{"role": "user", "content": prompt}]


        # response = self.client.chat.completions.create(
        response = self.client.completions.create(
            prompt=prompt,
            model=self.engine,
            # messages=messages,
            max_tokens=self.max_tokens,
            n=self.n,
            logprobs=self.logprobs,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop,
        )

        output_seq_score = []
        for index, choice in enumerate(response.choices):
            if choice.logprobs and choice.logprobs.token_logprobs:
                probs = []
                for prob, tok in zip(choice.logprobs.token_logprobs, choice.logprobs.tokens):
                    if tok not in self.stop and tok != "<|endoftext|>":
                        probs.append(prob)
                    else:
                        probs.append(prob)
                        break

                score = -sum(probs) / len(probs) if probs else 100.0
                output_seq_score.append((choice.text, score))
            else:
                output_seq_score.append((choice.text, index))



        # output_seq_score = []
        # for index, choice in enumerate(response.choices):
        #     if choice.logprobs and choice.logprobs.token_logprobs:
        #         probs = []
        #         for prob, tok in zip(choice.logprobs.token_logprobs, choice.logprobs.tokens):
        #             if tok not in self.stop and tok != "<|endoftext|>":
        #                 probs.append(prob)
        #             else:
        #                 probs.append(prob)
        #                 break

        #         score = -sum(probs) / len(probs) if probs else 100.0
        #         output_seq_score.append((choice.message.content, score))
        #     else:
        #         output_seq_score.append((choice.message.content, index))

        return sorted(output_seq_score, key=lambda x: x[1])
