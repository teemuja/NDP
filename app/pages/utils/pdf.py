# pdf plotters
import plotly.io as pio
import io
import base64
from fpdf import FPDF

#pdf https://stackoverflow.com/questions/47195075/insert-base64-image-to-pdf-using-pyfpdf
def generate_pdf_report(figure):
    # Convert the Plotly figure to a static image in memory
    image_str = pio.to_image(figure, format='png')
    #f = image_str.split('base64,')[1]
    f = base64.b64decode(image_str)
    f = io.BytesIO(f)
    # Create a new PDF document
    pdf = FPDF('P', 'mm', 'A4')
    # Add the image to the PDF document
    pdf.add_page()
    pdf.image('images/qissalogo.png', x=10, y=10, w=50, h=50)
    pdf.image(f, x=10, y=20, w=190, h=277)
    # Save the PDF document to bytes
    byte_string = pdf.output(dest='S').encode('latin-1')
    return byte_string
