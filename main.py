from model.nets import Generator

from tkinter import *
from tkinter import filedialog, font
from PIL import Image, ImageTk, ImageOps

import torch
import PIL
import torchvision.transforms as tt

Weights_path = 'model/generator_weights.pt'

def get_generated(x, g, f=tt.ToPILImage(), origin_size=[256, 256] ):
    generated = g(transforms(x)[None, :])
    with torch.no_grad():
        generated = generated.cpu().view(3, 256, 256)*0.5+0.5 
    return f(generated).resize(origin_size)
    
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
transforms = tt.Compose( [
                          tt.Resize([256, 256]),
                          tt.ToTensor(),
                          tt.Normalize(mean=mean, std=std)
 ] )

Generator_x = Generator()
Generator_z = Generator()

checkpoint = torch.load(Weights_path)
Generator_z.load_state_dict(checkpoint['gz'])
Generator_x.load_state_dict(checkpoint['gx'])

Generator_x.eval()
Generator_z.eval()
     
root = Tk()      
root.resizable(False, False)
root.iconbitmap('images/Cyfra.ico')
root.title('AniGAN')
WIDTH = 800
HEIGHT = 450
canvas = Canvas(root, width = WIDTH, height = HEIGHT, bg='white')      
canvas.pack()     

def save_output_1():
    filename = filedialog.asksaveasfile(mode='wb', defaultextension=".png")
    if not filename:
        return
    img1_out_raw.save(filename)

def open_img_1():
    global img1, img1_out, img1_out_raw, btn_img1_save
    try:
        btn_img1_save.place_forget()
    except NameError:
        pass
    try:
        btn_img1.place_forget()
        btn_img1_out.place_forget()
    except UnboundLocalError:
        pass
    
    input1 = filedialog.askopenfilename(title='choose a face')
    
    img1 = Image.open(input1).convert('RGB')
    origin_size = img1.size
    img1 = img1.resize( (256, 256)) #, Image.ANTIALIAS 
    img1_out_raw = get_generated(img1, Generator_x, origin_size=origin_size)
    img1_out = ImageTk.PhotoImage(get_generated(img1, Generator_x), master=root)
    img1 = ImageTk.PhotoImage(img1, master=root) 
    canvas.image = img1
    canvas.image = img1_out
    btn1.place_forget()
    
    btn_img1_save = Button(root, text='Save output', command=save_output_1)
    btn_img1_save.place(x=WIDTH - WIDTH//5 -25, y=HEIGHT-100)
    
    btn_img1 = Button(root, image=img1, command=open_img_1)
    btn_img1_out = Button(root, image=img1_out, command=ignore)
    btn_img1.place(x=25, y=50)
    btn_img1_out.place(x=WIDTH - WIDTH//5-256/2, y=50)
    canvas.pack()
    
    
button_img = Image.open("images/Button_sample.png").resize( (256, 256))
button_img = ImageTk.PhotoImage(button_img, master=root) 
out_img = Image.open("images/doggo.png").resize( (256, 256))
out_img = ImageTk.PhotoImage(out_img, master=root) 
btn1 = Button(root, image=button_img, command=open_img_1)
def ignore(): pass
output = Button(root, image=out_img, command=ignore)

btn1.place(x=25, y=50)
output.place(x=WIDTH - WIDTH//5-256/2, y=50)


img3 = Image.open("images/Arrow.jpg")
img3 = img3.rotate(-90, fillcolor='white', expand=True).resize( (WIDTH//4, HEIGHT//4) )
img3 = ImageTk.PhotoImage(img3, master=root) 


canvas.create_text((WIDTH - WIDTH//5, 30), text='Your output here', font=font.Font(family='Helvetica',size=16))
canvas.create_text((WIDTH//5, 30), text='Real face to Anime', font=font.Font(family='Helvetica',size=16))

def Open():
    top = Toplevel()
    # Add a label to the TopLevel, just like you would the root window
    Instruction_img = ImageTk.PhotoImage(Image.open("images/Instruction.png"), master=root) 
    canvas.image = Instruction_img
    lbl = Label(top, image=Instruction_img)
    lbl.pack()
Instruction_btn = Button(root, text="?", font=font.Font(family='Helvetica',size=16), command=Open)
Instruction_btn.place(x=WIDTH/2, y=HEIGHT-50)

canvas.create_image(WIDTH//2, HEIGHT/2.5, image=img3)            

mainloop()

