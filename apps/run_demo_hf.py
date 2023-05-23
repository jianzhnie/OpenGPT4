import argparse

import gradio as gr
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration


def gradio_demo():
    image_input = gr.Image(type='pil')

    min_len = gr.Slider(
        minimum=1,
        maximum=50,
        value=1,
        step=1,
        interactive=True,
        label='Min Length',
    )

    max_len = gr.Slider(
        minimum=10,
        maximum=500,
        value=250,
        step=5,
        interactive=True,
        label='Max Length',
    )
    top_p = gr.Slider(
        minimum=0.5,
        maximum=1.0,
        value=0.9,
        step=0.1,
        interactive=True,
        label='Top p',
    )

    beam_size = gr.Slider(
        minimum=1,
        maximum=10,
        value=5,
        step=1,
        interactive=True,
        label='Beam Size',
    )

    len_penalty = gr.Slider(
        minimum=-1,
        maximum=2,
        value=1,
        step=0.2,
        interactive=True,
        label='Length Penalty',
    )

    repetition_penalty = gr.Slider(
        minimum=-1,
        maximum=3,
        value=1,
        step=0.2,
        interactive=True,
        label='Repetition Penalty',
    )

    prompt_textbox = gr.Textbox(label='Prompt:', placeholder='prompt', lines=2)

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print('Loading model...')
    processor = AutoProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-opt-2.7b', torch_dtype=torch.float16)

    print('Loading model done!')

    def inference(image, prompt, min_len, max_len, beam_size, len_penalty,
                  repetition_penalty, top_p):

        inputs = processor(image, text=prompt,
                           return_tensors='pt').to(device, torch.float16)

        generated_ids = model.generate(**inputs,
                                       length_penalty=float(len_penalty),
                                       repetition_penalty=float(
                                           repetition_penalty),
                                       num_beams=beam_size,
                                       max_length=max_len,
                                       min_length=min_len,
                                       top_p=top_p,
                                       max_new_tokens=20)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text

    demo = gr.Interface(
        fn=inference,
        inputs=[
            image_input, prompt_textbox, min_len, max_len, beam_size,
            len_penalty, repetition_penalty, top_p
        ],
        outputs='text',
        allow_flagging='never',
    )
    demo.launch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--model-name', default='blip2_vicuna_instruct')
    parser.add_argument('--model-type', default='vicuna7b')
    args = parser.parse_args()
    gradio_demo()
