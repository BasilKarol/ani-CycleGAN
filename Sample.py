from tkinter import *
from tkinter import filedialog, font
from PIL import Image, ImageTk, ImageOps

import torch
import torch.nn as nn
import numpy as np
import PIL
import torchvision.transforms as tt
Path = 'AnimeGAN+_70_params.pt'

def get_generated(x, g, f=tt.ToPILImage(), origin_size=[256, 256] ):
    generated = g(transforms(x)[np.newaxis, :])
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
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1,bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
            )
        
        self.res_block_sample = [
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.InstanceNorm2d(256)
        ]
        self.Res_block1 = nn.Sequential(*self.res_block_sample)
        self.Res_block2 = nn.Sequential(*self.res_block_sample)
        self.Res_block3 = nn.Sequential(*self.res_block_sample)
        self.Res_block4 = nn.Sequential(*self.res_block_sample)
        self.Res_block5 = nn.Sequential(*self.res_block_sample)
        self.Res_block6 = nn.Sequential(*self.res_block_sample)
        self.Res_block7 = nn.Sequential(*self.res_block_sample)
        self.Res_block8 = nn.Sequential(*self.res_block_sample)
        self.Res_block9 = nn.Sequential(*self.res_block_sample)
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.downsample(x)
        
        
        #9 times for 256*256, 6 times for 128*128
        x = x + self.Res_block1(x)
        x = x + self.Res_block2(x)
        x = x + self.Res_block3(x)
        x = x + self.Res_block4(x)
        x = x + self.Res_block5(x)
        x = x + self.Res_block6(x)
        x = x + self.Res_block7(x)
        x = x + self.Res_block8(x)
        x = x + self.Res_block9(x)
        
        x = self.upsample(x)
        return x
    
Generator_x = Generator()
Generator_z = Generator()

checkpoint = torch.load(Path)
Generator_z.load_state_dict(checkpoint['gz'])
Generator_x.load_state_dict(checkpoint['gx'])

Generator_x.eval()
Generator_z.eval()
     


root = Tk()      
root.resizable(False, False)
root.iconbitmap('Cyfra.ico')
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
def save_output_2():
    filename = filedialog.asksaveasfile(mode='wb', defaultextension=".png")
    if not filename:
        return
    img2_out_raw.save(filename)

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
    #try:
    #    canvas.image = img2
    #    canvas.image = img2_out
    #    btn_img2_save.place_forget()
    #except NameError:
    #    pass
    #create button
    btn1.place_forget()
    
    btn_img1_save = Button(root, text='Save output', command=save_output_1)
    btn_img1_save.place(x=WIDTH - WIDTH//5 -25, y=HEIGHT-100)
    
    btn_img1 = Button(root, image=img1, command=open_img_1)
    btn_img1_out = Button(root, image=img1_out, command=ignore)
    btn_img1.place(x=25, y=50)
    btn_img1_out.place(x=WIDTH - WIDTH//5-256/2, y=50)
    canvas.pack()
    
def open_img_2():
    global img2, img2_out, img2_out_raw, btn_img2_save
    try:
        btn_img2_save.place_forget()
    except NameError:
        pass
    try:
        btn_img2.place_forget()
        btn_img2_out.place_forget()
    except UnboundLocalError:
        pass
    input2 = filedialog.askopenfilename(title='choose a face')
    
    img2 = Image.open(input2).convert('RGB')
    origin_size = img2.size
    img2 = img2.resize( (256, 256)) #, Image.ANTIALIAS 
    img2_out_raw = get_generated(img2, Generator_z, origin_size=origin_size)
    img2_out = ImageTk.PhotoImage(get_generated(img2, Generator_z), master=root)
    img2 = ImageTk.PhotoImage(img2, master=root) 
    try:
        canvas.image = img1
        canvas.image = img1_out
        btn_img1_save.place_forget()
    except NameError:
        pass
    canvas.image = img2
    canvas.image = img2_out
    #create button
    btn2.place_forget()
    
    btn_img2_save = Button(root, text='Save output', command=save_output_2)
    btn_img2_save.place(x=WIDTH//5, y=HEIGHT-100)
    
    btn_img2 = Button(root, image=img2, command=open_img_2)
    btn_img2_out = Button(root, image=img2_out, command=open_img_1)
    btn_img2.place(x=WIDTH - WIDTH//5-256/2, y=50)
    btn_img2_out.place(x=25, y=50)
    canvas.pack()
    
    
button_img = Image.open("Button_sample.png").resize( (256, 256))
button_img = ImageTk.PhotoImage(button_img, master=root) 
out_img = Image.open("doggo.png").resize( (256, 256))
out_img = ImageTk.PhotoImage(out_img, master=root) 
btn1 = Button(root, image=button_img, command=open_img_1)
def ignore(): pass
output = Button(root, image=out_img, command=ignore)

btn1.place(x=25, y=50)
output.place(x=WIDTH - WIDTH//5-256/2, y=50)


img3 = Image.open("Arrow.jpg")
#img3_1 = Image.open("Arrow.jpg")
img3 = img3.rotate(-90, fillcolor='white', expand=True).resize( (WIDTH//4, HEIGHT//4) )
#img3_1 = img3_1.rotate(90, fillcolor='white', expand=True).resize( (WIDTH//4, HEIGHT//8) )
img3 = ImageTk.PhotoImage(img3, master=root) 
#img3_1 = ImageTk.PhotoImage(img3_1, master=root) 


canvas.create_text((WIDTH - WIDTH//5, 30), text='Your output here', font=font.Font(family='Helvetica',size=16))
canvas.create_text((WIDTH//5, 30), text='Real face to Anime', font=font.Font(family='Helvetica',size=16))

def Open():
    top = Toplevel()
    # Add a label to the TopLevel, just like you would the root window
    Instruction_img = ImageTk.PhotoImage(Image.open("Instruction.png"), master=root) 
    canvas.image = Instruction_img
    lbl = Label(top, image=Instruction_img)
    lbl.pack()
Instruction_btn = Button(root, text="?", font=font.Font(family='Helvetica',size=16), command=Open)
Instruction_btn.place(x=WIDTH/2, y=HEIGHT-50)

canvas.create_image(WIDTH//2, HEIGHT/2.5, image=img3)            
#canvas.create_image(WIDTH//2, HEIGHT/2.5-(HEIGHT//16), image=img3_1) 

mainloop()

