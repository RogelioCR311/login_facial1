import cv2
import face_recognition as fr
import numpy as np
import mediapipe as mp
import os
from tkinter import *
from PIL import Image, ImageTk
import imutils
import math
from env import PATH
import time

# Paths
outputFolderPathUser = f'{PATH}/database/users'
pathUserCheck = f'{PATH}/database/users/'
outputFolderPathFace = f'{PATH}/database/faces'

def profile():
    global step, conteo, username, outputFolderPathUser

    step = 0
    conteo = 0

    # Window
    pantalla4 = Toplevel(pantalla)
    pantalla4.title('PROFILE')
    pantalla4.geometry("1280x720")

    # Background
    bc = Label(pantalla4, image=imagenbc, text='Bienvenido')
    bc.place(x=0, y=0, relheight=1, relwidth=1)

    # File
    userFile = open(f'{outputFolderPathUser}/{username}.txt', 'r')
    infoUser = userFile.read().split(',')
    name = infoUser[0]
    user = infoUser[1]
    passwd = infoUser[2]

    if user in clases:
        texto1 = Label(pantalla4, text=f'BIENVENIDO {name}')
        texto1.place(x=580, y=50)

        # label Img
        lblimage = Label(pantalla4)
        lblimage.place(x=490, y=80)

        imgUser = cv2.imread(f'{outputFolderPathFace}/{user}.png')
        imgUser = cv2.cvtColor(imgUser, cv2.COLOR_RGB2BGR)
        imgUser = Image.fromarray(imgUser)

        IMG = ImageTk.PhotoImage(image=imgUser)

        lblimage.configure(image=IMG)
        lblimage.image = IMG

def codeFace(images):
    listacod = []

    # Iteramos
    for img in images:
        # Color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # IMG Code
        cod = fr.face_encodings(img)[0]
        # Save List
        listacod.append(cod)
    
    return listacod

def closeWindow():
    global step, conteo
    conteo = 0
    step = 0
    pantalla2.destroy()

def closeWindow2():
    global step, conteo
    conteo = 0
    step = 0
    pantalla3.destroy()

def signUpBiometric():
    global pantalla2, conteo, parpadeo, img_info, step, cap, lblVideo

    # Check Cap
    if cap is not None:
        ret, frame = cap.read()

        # Resize
        frame = imutils.resize(frame, width=1280)

        # RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameSave = frame.copy()

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

                            # Parietal derecho
                            x5, y5 = lista[139][1:]

                            #Parietal izquierdo
                            x6, y6 = lista[368][1:]

                            # Ceja derecha
                            x7, y7 = lista[70][1:]
                            x8, y8 = lista[300][1:]


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
                                        xf = xi + anc

                                        # Offset Y
                                        offsetal = (offsety / 100 ) * alt
                                        yi = int(yi - int(offsetal/2))
                                        alt = int(alt + offsetal)
                                        yf = yi + alt
                                        
                                        # Error
                                        if xi < 0 : xi = 0
                                        if yi < 0 : yi = 0
                                        if anc < 0 : anc = 0
                                        if alt < 0 : alt = 0
                                        
                                        # Steps
                                        if step == 0:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (255, 255, 255), 2)

                                            # IMG Step0
                                            als0, ans0, c = img_step0.shape
                                            frame[50:50 + als0, 50:50 + ans0] = img_step0
                                            # IMG Step1
                                            als1, ans1, c = img_step1.shape
                                            frame[50:50 + als1, 1030:1030 + ans1] = img_step1
                                            # IMG Step2
                                            als2, ans2, c = img_step2.shape
                                            frame[270:270 + als2, 1030:1030 + ans2] = img_step2

                                            # Face Center
                                            if x7 > x5 and x8 < x6:
                                                #IMG Check
                                                # IMG Step2
                                                alch, anch, c = img_check.shape
                                                frame[165:165 + alch, 1105:1105 + anch] = img_check
                                                # Conteo Parpadeo
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True
                                                elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:
                                                    parpadeo = False

                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 375), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255,255), 1)
                                                
                                                if conteo >= 3:
                                                    # IMG Check
                                                    alch, anch, c = img_check.shape
                                                    frame[385:385 + alch, 1105:1105 + anch] = img_check

                                                    # Open Eyes
                                                    if longitud1 > 11 and longitud2 > 11:
                                                        cut = frameSave[yi:yf, xi:xf]

                                                        # Save
                                                        cv2.imwrite(f'{outputFolderPathFace}/{regUser}.png', cut)

                                                        step = 1

                                            else:
                                                conteo = 0
                                        
                                        if step == 1:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (0, 255, 0), 2)
                                            # IMG Check Liveness
                                            alli, anli, c = img_liche.shape
                                            frame[50:50 + alli, 50:50 + anli] = img_liche
                                            time.sleep(7)
                                            closeWindow()
                                            return

        # Convert video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Show video
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, signUpBiometric)
    
    else:
        cap.release()

def signInBiometric():
    global logUser, logPass, outFolderPath, cap, lblVideo, pantalla3, faceCode, clases, images, step, parpadeo, conteo, username

        # Check Cap
    if cap is not None:
        ret, frame = cap.read()

        # Resize
        frame = imutils.resize(frame, width=1280)

        # RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameSave = frame.copy()

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

                            # Parietal derecho
                            x5, y5 = lista[139][1:]

                            #Parietal izquierdo
                            x6, y6 = lista[368][1:]

                            # Ceja derecha
                            x7, y7 = lista[70][1:]
                            x8, y8 = lista[300][1:]


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
                                        xf = xi + anc

                                        # Offset Y
                                        offsetal = (offsety / 100 ) * alt
                                        yi = int(yi - int(offsetal/2))
                                        alt = int(alt + offsetal)
                                        yf = yi + alt
                                        
                                        # Error
                                        if xi < 0 : xi = 0
                                        if yi < 0 : yi = 0
                                        if anc < 0 : anc = 0
                                        if alt < 0 : alt = 0
                                        
                                        # Steps
                                        if step == 0:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (255, 255, 255), 2)

                                            # IMG Step0
                                            als0, ans0, c = img_step0.shape
                                            frame[50:50 + als0, 50:50 + ans0] = img_step0
                                            # IMG Step1
                                            als1, ans1, c = img_step1.shape
                                            frame[50:50 + als1, 1030:1030 + ans1] = img_step1
                                            # IMG Step2
                                            als2, ans2, c = img_step2.shape
                                            frame[270:270 + als2, 1030:1030 + ans2] = img_step2

                                            # Face Center
                                            if x7 > x5 and x8 < x6:
                                                #IMG Check
                                                # IMG Step2
                                                alch, anch, c = img_check.shape
                                                frame[165:165 + alch, 1105:1105 + anch] = img_check
                                                # Conteo Parpadeo
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True
                                                elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:
                                                    parpadeo = False

                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 375), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255,255), 1)
                                                
                                                if conteo >= 3:
                                                    # IMG Check
                                                    alch, anch, c = img_check.shape
                                                    frame[385:385 + alch, 1105:1105 + anch] = img_check

                                                    # Open Eyes
                                                    if longitud1 > 11 and longitud2 > 11:
                                                        step = 1

                                            else:
                                                conteo = 0
                                        
                                        if step == 1:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (0, 255, 0), 2)
                                            # IMG Check Liveness
                                            alli, anli, c = img_liche.shape
                                            frame[50:50 + alli, 50:50 + anli] = img_liche
                                            
                                            # Find Faces
                                            facess = fr.face_locations(frameRGB)
                                            facescod = fr.face_encodings(frameRGB, facess)

                                            # Iteramos
                                            for facecod, facesloc in zip(facescod, facess):

                                                # Matching
                                                match = fr.compare_faces(faceCode, facecod)

                                                # Sim
                                                simi = fr.face_distance(faceCode, facecod)

                                                # Min
                                                min = np.argmin(simi)

                                                if match[min]:
                                                    username = clases[min].upper()

                                                    profile()

                                            closeWindow2()
                                            return

        # Convert video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Show video
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, signInBiometric)
    
    else:
        cap.release()

def signUp():
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
            signUpBiometric()

def signIn():
    global logUser, logPass, outputFolderPathFace, cap, lblVideo, pantalla3, faceCode, clases, images

    logUser, logPass = inputUserLog.get(), inputPassLog.get()

    # DB Faces
    images = []
    clases = []
    lista = os.listdir(outputFolderPathFace)

    for lis in lista:
        # Read Img
        imgdb = cv2.imread(f'{outputFolderPathFace}/{lis}')
        images.append(imgdb)
        clases.append(os.path.splitext(lis)[0])

    faceCode = codeFace(images)

    # Window
    pantalla3 = Toplevel(pantalla)
    pantalla3.title('BIOMETRIC SIGN IN')
    pantalla3.geometry("1280x720")

    # Label Video
    lblVideo = Label(pantalla3)
    lblVideo.place(x=0, y=0)

    # Videocaptura
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    signInBiometric()


# Read img
img_info = cv2.imread(f'{PATH}/setup/Info.png')
img_check = cv2.imread(f'{PATH}/setup/check.png')
img_step0 = cv2.imread(f'{PATH}/setup/Step0.png')
img_step1 = cv2.imread(f'{PATH}/setup/Step1.png')
img_step2 = cv2.imread(f'{PATH}/setup/Step2.png')
img_liche = cv2.imread(f'{PATH}/setup/LivenessCheck.png')

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
imagenF = PhotoImage(file=f'{PATH}/setup/Inicio.png')
background = Label(image=imagenF, text='Inicio')
background.place(x=0, y=0, relheight=1, relwidth=1)

# BG pantalla inicio
imagenbc = PhotoImage(file=f'{PATH}/setup/Back2.png')

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
inputPassLog = Entry(pantalla)
inputPassLog.place(x=730, y=530)

# Button
# SignUp
imagenBR = PhotoImage(file=f'{PATH}/setup/BtSign.png')
btReg = Button(pantalla, text='Registro', image=imagenBR, height='40', width='200', command=signUp)
btReg.place(x=300, y=580)

# LogIn 
imagenBL = PhotoImage(file=f'{PATH}/setup/BtLogin.png')
btLog = Button(pantalla, text='Inicio', image=imagenBL, height='40', width='200', command=signIn)
btLog.place(x=800, y=580)

pantalla.mainloop()