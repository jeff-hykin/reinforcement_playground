
def show(*, image_tensor, format='png'):
    from tools.pytorch_tools import tensor_to_image
    import IPython.display
    from io import BytesIO
    buffer = BytesIO()
    image = tensor_to_image(image_tensor)
    # save to stream
    image.save(buffer, fmt)
    # display stream as if it was a file
    IPython.display.display(IPython.display.Image(data=buffer.getvalue()))