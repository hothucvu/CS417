import gradio as gr
from predict import predict_clothing


def run_ui(image):
    label, confidence, prob_dict = predict_clothing(image)
    tag_text = f"[{label}]  —  Độ tin cậy: {confidence}%"
    return tag_text, prob_dict


with gr.Blocks(title="Phân loại sản phẩm thời trang (CS 417)") as demo:
    gr.Markdown("## Phân loại loại sản phẩm quần áo (Local Demo)")
    gr.Markdown("Tải lên 1 hình ảnh sản phẩm để AI dự đoán loại sản phẩm.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Ảnh sản phẩm")
        with gr.Column():
            tag_output = gr.Textbox(label="Thẻ kết quả", interactive=False)
            prob_output = gr.Label(label="Xác suất từng loại", num_top_classes=5)

    btn = gr.Button("Dự đoán", variant="primary")
    btn.click(fn=run_ui, inputs=image_input, outputs=[tag_output, prob_output])
    image_input.change(fn=run_ui, inputs=image_input, outputs=[tag_output, prob_output])


if __name__ == "__main__":
    demo.launch()