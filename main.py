import cv2
import face_recognition as fr
import numpy as np
import mediapipe as mp
import os
from tkinter import *
from PIL import Image, ImageTk
import imutils
import math

# Paths
outputFolderPathUser = 'C:/Users/Rogelio/Desktop/facial_login/database/users'
pathUserCheck = 'C:/Users/Rogelio/Desktop/facial_login/database/users/'
outputFolderPathFace = 'C:/Users/Rogelio/Desktop/facial_login/database/faces'

def signBiometric():
    global pantalla2, conteo, parpadeo, img_info, step, cap, lblVideo

    # Check Cap
    if cap is not None:
        ret, frame = cap.read()

        # Resize
        frame = imutils.resize(frame, width=1280)

        # RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Frame show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if ret == True:
            # Interference Face Mesh
            res = faceMesh.process(frameRGB)

            # Result List
            px = []
            py = []
            lista = []

            if res.multi_face_landmarks:
                # Extract Face Mesh
                for rostros in res.multi_face_landmarks:
                    # Draw
                    mpDraw.draw_landmarks(frame, rostros, faceMeshObject.FACEMESH_CONTOURS, configDraw, configDraw)

                    # Extract KeyPoint
                    for id, puntos in enumerate(rostros.landmark):
                        # Info img
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        # 468 KeyPoints
                        if len(lista) == 468:
                            # Ojo derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            longitud1 = math.hypot(x2-x1, y2-y1)
                            
                            # Ojo derecho
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            longitud2 = math.hypot(x4-x3, y4-y3)

                            #Face Detect
                            faces = detector.process(frameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:
                                    # Bbox: "ID, BBOX, SCORE"
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    # Threshold
                                    if score > confThreshold:
                                        # Pixels
                                        xi, yi, anc, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, anc, alt = int(xi * an), int(yi * al), int(anc * an), int(alt * al)

                                        # Offset X
                                        offsetan = (offsetx / 100 ) * anc
                                        xi = int(xi - int(offsetan/2))
                                        anc = int(anc + offsetan)
                                        
                                        # Offset Y
                                        offsetal = (offsety / 100 ) * alt
                                        yi = int(yi - int(offsetal/2))
                                        alt = int(alt + offsetal)
                                        
                                        # Error
                                        if xi < 0 : xi = 0
                                        if yi < 0 : yi = 0
                                        if anc < 0 : anc = 0
                                        if alt < 0 : alt = 0
                                                                            
                                        # Draw
                                        cv2.rectangle(frame, (xi, yi, anc, alt), (255, 255, 255), 2)

                            #Circle
                            cv2.circle(frame, (x1,y1), 2, (255,0,0), cv2.FILLED)
                            cv2.circle(frame, (x2,y2), 2, (255,0,0), cv2.FILLED)

        # Convert video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Show video
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, signBiometric)
    
    else:
        cap.release()

def sign():
    global regName, regUser, regPass, inputNameReg, inputUserReg, inputPassReg, cap, lblVideo, pantalla2
    # Extract: Name - User - Password
    regName, regUser, regPass = inputNameReg.get(), inputUserReg.get(), inputPassReg.get()

    # Form Incomplete
    if len(regName) == 0 or len(regUser) == 0 or len(regPass) == 0:
        print('FORMULARIO INCOMPLETO')
    else:
        userList = os.listdir(pathUserCheck)
        
        username = []

        # Check user List
        for lis in userList:
            #Extract User
            user = lis
            user = user.split('.')
            # Save User
            username.append(user[0])

        # Check User
        if regUser in username:
            print('USUARIO REGISTRADO ANTERIORMENTE')

        else:
            # No registred
            info.append(regName)
            info.append(regUser)
            info.append(regPass)

            # Export info
            f = open(f'{outputFolderPathUser}/{regUser}.txt', 'w')
            f.write(f'{regName},')
            f.write(f'{regUser},')
            f.write(regPass)
            f.close()

            inputNameReg.delete(0, END)
            inputUserReg.delete(0, END)
            inputPassReg.delete(0, END)

            # New screen
            pantalla2 = Toplevel(pantalla)
            pantalla2.title('LOGIN BIOMETRIC')
            pantalla2.geometry('1280x720')

            # Label Video
            lblVideo = Label(pantalla2)
            lblVideo.place(x=0, y=0)

            # Videocaptura
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(3, 1280)
            cap.set(4, 720)
            signBiometric()


def log():
    print('Hello')

# Variables
parpadeo = False
conteo = 0
muestra = 0
step = 0

# Offset
offsety = 30
offsetx = 20

# Threshold
confThreshold = 0.5

# Tool Draw
mpDraw = mp.solutions.drawing_utils
configDraw = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Object Face Mesh
faceMeshObject = mp.solutions.face_mesh
faceMesh = faceMeshObject.FaceMesh(max_num_faces=1)

# Object Detect
faceObject = mp.solutions.face_detection
detector = faceObject.FaceDetection(min_detection_confidence= 0.5, model_selection=1)

# Info List
info = []

# Ventana principal
pantalla = Tk()
pantalla.title('FACE RECOGNITION SYSTEM')
pantalla.geometry('1280x720')

#Fondo
imagenF = PhotoImage(file='C:/Users/Rogelio/Desktop/facial_login/setup/Inicio.png')
background = Label(image=imagenF, text='Inicio')
background.place(x=0, y=0, relheight=1, relwidth=1)

# Input text
# Name
inputNameReg = Entry(pantalla)
inputNameReg.place(x=110, y=320)

# User
inputUserReg = Entry(pantalla)
inputUserReg.place(x=110, y=430)

# Pass
inputPassReg = Entry(pantalla)
inputPassReg.place(x=110, y=540)

# Input Text Sign Up
# User
inputUserLog = Entry(pantalla)
inputUserLog.place(x=730, y=420)

# Pass
inputUserLog = Entry(pantalla)
inputUserLog.place(x=730, y=530)

# Button
# Sign
imagenBR = PhotoImage(file='C:/Users/Rogelio/Desktop/facial_login/setup/BtSign.png')
btReg = Button(pantalla, text='Registro', image=imagenBR, height='40', width='200', command=sign)
btReg.place(x=300, y=580)

# Sign 
imagenBL = PhotoImage(file='C:/Users/Rogelio/Desktop/facial_login/setup/BtLogin.png')
btLog = Button(pantalla, text='Inicio', image=imagenBL, height='40', width='200', command=log)
btLog.place(x=800, y=580)

pantalla.mainloop()