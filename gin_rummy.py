import tensorflow as tf
import cv2
import numpy as np
import os
import pygame
import time
import random
from collections import deque, Counter

DEBUG = True
MODEL_PATH = "64x3-cards.h5"
TRAIN_FOLDER = r"D:\TUGAS KULIAHHH\SEM 5\12 2024\GIn Rummy Game\Image Label"
CARD_IMAGES_FOLDER = r"D:\TUGAS KULIAHHH\SEM 5\12 2024\GIn Rummy Game\Image Thumbnail"
DETECTED_CARDS_FILE = "detected_card.txt"
MIN_AREA = 5000
FRAME_BUFFER_SIZE = 90
CARD_WIDTH = 138
CARD_HEIGHT = 211
CARD_SPACING = 10
detecting_cards = True
read_bot = False
key_held = False

UIfont="Roboto-Bold.ttf"

virtual_deck_list=[]
set_list = [] 
run_list = []
detected_cards_file = []  
last_detected_cards = [] 
discarded_cards = []     
drawn_cards = []         
drawn_card=[]
decks = pygame.image.load(r"D:\TUGAS KULIAHHH\SEM 5\12 2024\GIn Rummy Game\Image Thumbnail\deck.png")  
decks = pygame.transform.scale(decks, (250, 352))

bot_set_list = []  
bot_run_list = []  
bot_detected_cards = [] 
bot_last_detected_cards = []  
bot_drawn_cards = []  
total_score = 0
bot_total = 0
model = tf.keras.models.load_model(MODEL_PATH)
classes = [i for i in os.listdir(TRAIN_FOLDER) if os.path.isdir(os.path.join(TRAIN_FOLDER, i))]

frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

def debug_print(message):
    if DEBUG:
        print(f"[DEBUG]: {message}")

def prepare_card(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img.reshape(-1, 128, 128, 1)

def draw_circle(image, y, x):
    center_coor = (x, y)
    radius = 10
    color = (0, 0, 255)
    thickness = 2
    image = cv2.circle(image, center_coor, radius, color, thickness)
    return image

def force_vertical(box):
    width = int(np.linalg.norm(box[0] - box[1]))
    height = int(np.linalg.norm(box[1] - box[2]))
    if width > height:
        box = np.roll(box, 1, axis=0)
    return box

def save_top_cards():
    card_counts = Counter(frame_buffer)
    filtered_cards = [card for card, count in card_counts.items() if count >= 5]
    top_cards = sorted(filtered_cards, key=lambda x: -card_counts[x])[:10]
    
    with open("detected_card.txt", "w") as f:
        for card in top_cards:
            f.write(f"{card}\n")
    print(f"File detected_card.txt dibuat dengan kartu:\n" + "\n".join(top_cards))

def save_bot_card():
    print("dari bot")
    card_counts = Counter(frame_buffer)
    filtered_cards = [card for card, count in card_counts.items() if count >= 5]
    top_cards = sorted(filtered_cards, key=lambda x: -card_counts[x])[:10]
    
    with open("bot_gin.txt", "w") as f:
        for card in top_cards:
            f.write(f"{card}\n")
    print(f"File bot_gin.txt dibuat dengan kartu:\n" + "\n".join(top_cards))
    
def load_bot_cards():
    global bot_detected_cards
    try:
        with open("bot_gin.txt", "r") as f:
            bot_detected_cards = [line.strip() for line in f.readlines()]
            update_bot_deck(bot_detected_cards)

        print(f"Kartu bot yang terdeteksi: {', '.join(bot_detected_cards)}")
    except FileNotFoundError:
        print("File 'bot_gin.txt' tidak ditemukan!")



def virtual_deck():
    suits = ["clubs", "diamonds", "hearts", "spades"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    deck = [f"{rank} {suit}" for suit in suits for rank in ranks]
    random.shuffle(deck)  
    return deck

def remove_detected_cards(deck, detected_cards_file):
    removed_cards = []
    for card in detected_cards_file:
        if card in deck:
            deck.remove(card)
            removed_cards.append(card)
    if removed_cards:
        print(f"Kartu yang dihapus dari deck: {', '.join(removed_cards)}")
    print(f"Total kartu di virtual deck: {len(virtual_deck_list)}")
    return deck

def bot_remove_detected_cards(deck, bot_detected_cards):
    removed_cards = []
    for card in bot_detected_cards:
        if card in deck:
            deck.remove(card)
            removed_cards.append(card)
    if removed_cards:
        print(f"Kartu yang dihapus dari deck: {', '.join(removed_cards)}")
    print(f"Total kartu di virtual deck: {len(virtual_deck_list)}")
    return deck

def update_virtual_deck(detected_cards_file):
    global virtual_deck_list
    virtual_deck_list = remove_detected_cards(virtual_deck_list, detected_cards_file)
    
def update_bot_deck(bot_detected_cards):
    global virtual_deck_list
    virtual_deck_list = bot_remove_detected_cards(virtual_deck_list, bot_detected_cards)
    
def melds_run(detected_cards_file):
    global run_list  
    run_added = False 
    suit_cards = {
        "clubs": [],
        "diamonds": [],
        "hearts": [],
        "spades": []
    }
    for card in detected_cards_file:
        card_parts = card.split()  
        card_value = card_parts[0] 
        card_suit = card_parts[1]  
        if card_suit in suit_cards:
            suit_cards[card_suit].append(card_value)
        else:
            debug_print(f"Suit kartu tidak dikenali: {card_suit}")
    for suit, cards in suit_cards.items():
        if len(cards) >= 3: 
            card_values_numeric = {
                "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
                "J": 11, "Q": 12, "K": 13, "A": 14  
            }

            numeric_cards = sorted([card_values_numeric[value] for value in cards])
            for i in range(len(numeric_cards) - 2):  
                run_cards = []
                run_cards.append(f"{list(card_values_numeric.keys())[list(card_values_numeric.values()).index(numeric_cards[i])]} {suit}")
                for j in range(i + 1, len(numeric_cards)):
                    if numeric_cards[j] == numeric_cards[i] + 1:
                        run_cards.append(f"{list(card_values_numeric.keys())[list(card_values_numeric.values()).index(numeric_cards[j])]} {suit}")
                        i = j  
                    else:
                        break
                if len(run_cards) >= 3:
                    if not any(card in run_list for card in run_cards):
                        run_list.extend(run_cards) 
                        print(f"Run baru ditemukan: {run_cards}")  
                        run_added = True  
                        
    if run_added:
        print("Runs updated:", run_list)

def knock_settle(screen, total_score, bot_total, screen_width, screen_height):
    font = pygame.font.Font(None, 40)
    
    box_width = 400
    box_height = 200
    x_offset = (screen_width - box_width) // 2
    y_offset = (screen_height - box_height) // 2

    pygame.draw.rect(screen, (255, 255, 255), (x_offset, y_offset, box_width, box_height))

    total_score_text = f"Player Score: {total_score}"
    bot_score_text = f"Bot Score: {bot_total}"

    player_text_surface = font.render(total_score_text, True, (0, 0, 0))
    bot_text_surface = font.render(bot_score_text, True, (0, 0, 0))
    
    screen.blit(player_text_surface, (x_offset + (box_width - player_text_surface.get_width()) // 2, y_offset + 20))
    screen.blit(bot_text_surface, (x_offset + (box_width - bot_text_surface.get_width()) // 2, y_offset + 70))

    if total_score < bot_total:
        result_text = "Player Menang!"
    elif bot_total < total_score:
        result_text = "Bot Menang!"
    else:
        result_text = "Seri!"

    result_text_surface = font.render(result_text, True, (255, 0, 0))
    screen.blit(result_text_surface, (x_offset + (box_width - result_text_surface.get_width()) // 2, y_offset + 120))

    pygame.display.flip()

    time.sleep(5)
    pygame.quit()
    exit()

def bot_melds_run(bot_detected_cards):
    global bot_run_list
    run_added = False  
    suit_cards = {
        "clubs": [],
        "diamonds": [],
        "hearts": [],
        "spades": []
    }
    for card in bot_detected_cards:
        card_parts = card.split()  
        card_value = card_parts[0] 
        card_suit = card_parts[1]  
        if card_suit in suit_cards:
            suit_cards[card_suit].append(card_value)
    
    for suit, cards in suit_cards.items():
        if len(cards) >= 3:  
            card_values_numeric = {
                "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
                "J": 11, "Q": 12, "K": 13, "A": 14  
            }

            numeric_cards = sorted([card_values_numeric[value] for value in cards])
            for i in range(len(numeric_cards) - 2):  
                run_cards = []
                run_cards.append(f"{list(card_values_numeric.keys())[list(card_values_numeric.values()).index(numeric_cards[i])]} {suit}")
                for j in range(i + 1, len(numeric_cards)):
                    if numeric_cards[j] == numeric_cards[i] + 1:
                        run_cards.append(f"{list(card_values_numeric.keys())[list(card_values_numeric.values()).index(numeric_cards[j])]} {suit}")
                        i = j  
                    else:
                        break
                if len(run_cards) >= 3:
                    if not any(card in bot_run_list for card in run_cards):
                        bot_run_list.extend(run_cards) 
                        run_added = True  
    
    if run_added:
        print("Bot run updated:", bot_run_list)
        
def melds_set(detected_cards_file):
    global set_list  
    meld_added = False  
    rank_count = {}
    for card in detected_cards_file:
        card_parts = card.split() 
        card_value = card_parts[0] 

        if card_value in rank_count:
            rank_count[card_value].append(card)
        else:
            rank_count[card_value] = [card]
    
    for rank, cards in rank_count.items():
        if len(cards) >= 3:  
            if not any(card in set_list for card in cards[:3]):
                set_list.extend(cards[:3]) 
                print(f"Meld baru ditemukan: {cards[:3]}") 
                meld_added = True 
    if meld_added:
        print("Melds updated:", set_list)

def bot_melds_set(bot_detected_cards):
    global bot_set_list  
    meld_added = False  
    rank_count = {}
    for card in bot_detected_cards:
        card_parts = card.split() 
        card_value = card_parts[0] 

        if card_value in rank_count:
            rank_count[card_value].append(card)
        else:
            rank_count[card_value] = [card]
    
    for rank, cards in rank_count.items():
        if len(cards) >= 3:  
            if not any(card in bot_set_list for card in cards[:3]):
                bot_set_list.extend(cards[:3]) 
                meld_added = True  
    
    if meld_added:
        print("Bot melds updated:", bot_set_list)
        
def bot_draw_card(virtual_deck_list):
    global bot_drawn_cards
    if virtual_deck_list:
        drawn_card = virtual_deck_list.pop(0)  
        bot_drawn_cards.append(drawn_card)  
        print(f"Bot drew card: {drawn_card}")
        return drawn_card
    return None 

def load_card_images():
    card_images = {}
    for card_name in os.listdir(CARD_IMAGES_FOLDER):
        card_path = os.path.join(CARD_IMAGES_FOLDER, card_name)
        if os.path.isfile(card_path):
            try:
                card_base_name = os.path.splitext(card_name)[0]
                card_img = pygame.image.load(card_path)
                card_images[card_base_name] = card_img
            except pygame.error as e:
                debug_print(f"Failed to load {card_name}: {e}")
    return card_images

def calculate_score(detected_cards_file):
    global total_score  
    total_score = 0
    global set_list, run_list
    card_values = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
        "J": 10, "Q": 10, "K": 10, "A": 1  
    }
    all_melded_cards = set_list + run_list
    for card in detected_cards_file:
        card_parts = card.split()  
        card_value = card_parts[0]  

        if card not in all_melded_cards:
            if card_value in card_values:
                total_score += card_values[card_value]
            else:
                debug_print(f"Kartu tidak dikenali: {card}")
    return total_score

def bot_calculate_score(bot_detected_cards):
    global bot_total
    bot_total = 0
    card_values = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
        "J": 10, "Q": 10, "K": 10, "A": 1  
    }
    all_melded_cards = bot_set_list + bot_run_list
    for card in bot_detected_cards:
        card_parts = card.split()  
        card_value = card_parts[0]  

        if card not in all_melded_cards:
            if card_value in card_values:
                bot_total += card_values[card_value]
    return bot_total

def cek_gin(total_score, detected_cards_file, screen, screen_width, screen_height):
    if total_score == 0 and len(detected_cards_file) > 0:
        font = pygame.font.Font(None, 40)
        
        box_width = 300
        box_height = 150
        x_offset = (screen_width - box_width) // 2
        y_offset = (screen_height - box_height) // 2

        pygame.draw.rect(screen, (255, 255, 255), (x_offset, y_offset, box_width, box_height))

        gin_text = "Player Gin"
        gin_text_surface = font.render(gin_text, True, (0, 255, 0))
        screen.blit(gin_text_surface, (x_offset + (box_width - gin_text_surface.get_width()) // 2, y_offset + (box_height - gin_text_surface.get_height()) // 2))

        pygame.display.flip() 
        time.sleep(5)
        pygame.quit()
        exit()

        return True
    return False

def detect_cards(frame, cam):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 20, 40])
    upper = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    num_labels, labels_im = cv2.connectedComponents(mask)
    
    detected_classes = []

    for i in range(1, num_labels):
        b, k = np.where(labels_im == i)
        pts = np.column_stack((k, b))
        
        if len(pts) < MIN_AREA:
            continue

        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        box = force_vertical(box)

        dst_points = np.array([[0, 0], [230, 0], [230, 352], [0, 352]], dtype="float32")
        h_matrix = cv2.getPerspectiveTransform(np.float32(box), dst_points)
        straightened_card = cv2.warpPerspective(frame, h_matrix, (230, 352))

        card_for_model = prepare_card(straightened_card)
        prediction = model.predict(card_for_model)
        predicted_class_index = int(np.argmax(prediction))
        predicted_class = classes[predicted_class_index]

        frame_buffer.append(predicted_class)
        detected_classes.append(predicted_class)

        confidence = prediction[0][predicted_class_index] * 100  
        label = f"{predicted_class} ({confidence:.2f}%)"
        
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        for (x, y) in box:
            frame = draw_circle(frame, y, x)

        bottom_y = np.max(box[:, 1]) + 20 
        bottom_x = int(np.min(box[:, 0])) + 10

        cv2.putText(frame, label, (bottom_x, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, lineType=cv2.LINE_AA)

    return frame, detected_classes, mask
'''
def bot_logic():
    global bot_drawn_cards, bot_detected_cards, virtual_deck_list
    bot_drawn_cards.clear()
    for _ in range(8):
        bot_draw_card(virtual_deck_list)  
    bot_detected_cards.extend(bot_drawn_cards)
    print(f"Bot's detected cards: {', '.join(bot_detected_cards)}")
    
    '''
def bot_logic():
    global read_bot, key_held

    keys = pygame.key.get_pressed() 
    if keys[pygame.K_m] and not key_held: 
        read_bot = not read_bot  
        print(f"Tombol m ditekan: {read_bot}")  
        key_held = True 
    
    if not keys[pygame.K_m]:  
        key_held = False  

    return read_bot 

def bot_movement():
    global bot_detected_cards, virtual_deck_list, discarded_cards, drawn_card, bot_set_list, bot_run_list

    if discarded_cards:
        last_discarded_card = discarded_cards[-1]  
        temp_cards = bot_detected_cards + [last_discarded_card]


        valid_temp_cards = [
            card for card in temp_cards 
            if card not in bot_set_list and card not in bot_run_list
        ]
        #print("Kartu sementara:", valid_temp_cards)
        card_values = [card.split()[0] for card in valid_temp_cards]
        if any(card_values.count(value) >= 3 for value in set(card_values)): 
            bot_detected_cards.append(last_discarded_card)
            discarded_cards.pop()
            bot_melds_set(bot_detected_cards)
            bot_melds_run(bot_detected_cards)
            drawn_card = last_discarded_card
            update_bot_deck(bot_detected_cards)
            #print(f"Bot mengambil {last_discarded_card} dari discard pile untuk membuat meld set.")
            return  

        card_suits = {}
        card_values_numeric = {
            "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
            "J": 11, "Q": 12, "K": 13, "A": 14
        }

        for card in valid_temp_cards:
            value, suit = card.split()
            if suit not in card_suits:
                card_suits[suit] = []
            card_suits[suit].append(card_values_numeric[value])

        for suit, values in card_suits.items():
            values.sort()
            for i in range(len(values) - 2):
                if values[i] + 1 == values[i + 1] and values[i + 1] + 1 == values[i + 2]:
                    bot_detected_cards.append(last_discarded_card)
                    discarded_cards.pop()  
                    bot_melds_set(bot_detected_cards)
                    bot_melds_run(bot_detected_cards)
                    drawn_card = last_discarded_card
                    update_bot_deck(bot_detected_cards)
                    print(f"Bot mengambil {last_discarded_card} dari discard pile untuk membuat meld run.")
                    return  

    drawn_card = bot_draw_card(virtual_deck_list)
    if drawn_card:
        bot_detected_cards.append(drawn_card)
        bot_melds_set(bot_detected_cards)
        bot_melds_run(bot_detected_cards)
        update_bot_deck(bot_detected_cards)
        print(f"Bot menarik {drawn_card} dari deck.")
    else:
        print("Tidak ada kartu yang tersedia untuk diambil oleh bot.")



def bot_discard():
    global bot_detected_cards, discarded_cards, bot_set_list, bot_run_list

    if bot_detected_cards:
        valid_cards = [card for card in bot_detected_cards if card not in bot_set_list and card not in bot_run_list]
        
        if valid_cards:
            discarded_card = random.choice(valid_cards)
            bot_detected_cards.remove(discarded_card)
            discarded_cards.append(discarded_card)
            print(f"Bot membuang kartu: {discarded_card}")
        else:
            print("Bot tidak memiliki kartu yang dapat dibuang karena semua kartu di tangan ada di set atau run.")
    else:
        print("Bot tidak memiliki kartu untuk dibuang.")


def main():
    global detected_cards_file, last_detected_cards, discarded_cards, drawn_cards, virtual_deck_list
    pygame.init()
    info = pygame.display.Info()
    screen_width, screen_height =  info.current_w, info.current_h # 1280, 720

    screen = pygame.display.set_mode((screen_width, screen_height),pygame.FULLSCREEN)

    background_image = pygame.image.load("gincode.png") 
    background_image = pygame.transform.scale(background_image, (screen_width, screen_height)) 

    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("Error opening camera")
        exit()
        
    virtual_deck_list = virtual_deck()
     
    card_images = load_card_images()
    show_cards = True
    cards_loaded = False  
    card_remove = False
    card_load=False
    detecting_cards = True  
    botcard = "Scan kartu player dan bot lalu tekan Mulai"

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error in retrieving frame")
            break
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if detecting_cards:
            frame, detected_cards, mask = detect_cards(frame, cam) 

          
        frame = cv2.flip(frame, 1)
        if len(frame_buffer) == FRAME_BUFFER_SIZE:
            if read_bot:  
                print("membaca dari bot")
                save_bot_card()
                
            else: 
                save_top_cards()

            frame_buffer.clear() 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cam.release()
                pygame.quit()
                exit()
            bot_logic()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    detecting_cards = not detecting_cards  
                    status = "aktif" if detecting_cards else "non-aktif"
                    print(f"Deteksi kartu {status}.")
                elif event.key == pygame.K_k:  
                    cam.release()
                    pygame.quit()
                    break

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                mouse_pressed = pygame.mouse.get_pressed()

                if mouse_pressed[0] and (screen_width - 140 < mouse_x < screen_width - 20) and (screen_height // 2 - 60 < mouse_y < screen_height // 2 - 20):
                    print("Draw button clicked!")
                    if os.path.exists(DETECTED_CARDS_FILE):
                        with open(DETECTED_CARDS_FILE, "r") as file:
                            detected_cards_file = [line.strip() for line in file.readlines()]
                        debug_print(f"Detected cards from file: {detected_cards_file}")
                        
                        update_virtual_deck(detected_cards_file)
                        melds_set(detected_cards_file)
                        melds_run(detected_cards_file)
                
                        cards_to_remove_from_discarded = [card for card in discarded_cards if card in detected_cards_file]
                        
                        for card in cards_to_remove_from_discarded:
                            discarded_cards.remove(card)
                
                        if cards_to_remove_from_discarded:
                            print(f"Kartu yang diambil dari discarded cards dan dipindahkan ke tangan: {', '.join(cards_to_remove_from_discarded)}")
                
                        if detected_cards_file != last_detected_cards:
                            debug_print("Detected cards have changed. Updating display.")
                            
                            new_drawn = [card for card in detected_cards_file if card not in last_detected_cards]
                            drawn_cards.extend(new_drawn)
                
                            if new_drawn:
                                print(f"Kartu yang ditambah: {new_drawn}")
                
                            last_detected_cards = detected_cards_file
                        cards_loaded = True
                    else:
                        debug_print(f"File '{DETECTED_CARDS_FILE}' not found!")


                elif mouse_pressed[0] and (screen_width - 140 < mouse_x < screen_width - 20) and (screen_height // 2 + 20 < mouse_y < screen_height // 2 + 60):
                    print("Discard button clicked!")
                    if os.path.exists(DETECTED_CARDS_FILE):
                        with open(DETECTED_CARDS_FILE, "r") as file:
                            detected_cards_file = [line.strip() for line in file.readlines()]
                        debug_print(f"Detected cards from file: {detected_cards_file}")
                        update_virtual_deck(detected_cards_file)
                        melds_set(detected_cards_file)
                        melds_run(detected_cards_file)
                        if detected_cards_file != last_detected_cards:
                            debug_print("Detected cards have changed. Updating display.")
                            
                            new_discarded = [card for card in last_detected_cards if card not in detected_cards_file]
                            discarded_cards.extend(new_discarded)

                            if new_discarded:
                                print(f"Kartu yang dihapus: {new_discarded}")

                            last_detected_cards = detected_cards_file
                        
                        bot_movement()  
                        botcard = f"Bot mengambil {drawn_card}, buang dari deck fisik/discard pile lalu tekan konfirmasi"
                        card_remove = True
                    else:
                        debug_print(f"File '{DETECTED_CARDS_FILE}' not found!")
     
                elif mouse_pressed[0] and (screen_width - 140 < mouse_x < screen_width - 20) and (screen_height // 2 + 2 * (40 + 20) < mouse_y < screen_height // 2 + 2 * (40 + 20) + 40):
                    print("Konfirmasi Gerakan button clicked!")
                    bot_discard()  
                    botcard = f"Bot membuang {discarded_cards[-1]}, tambahkan ke discard pile"
                    
                    pygame.display.flip()  

                elif mouse_pressed[0] and (screen_width - 140 < mouse_x < screen_width - 20) and (screen_height // 2 + 3 * (40 + 20) < mouse_y < screen_height // 2 + 3 * (40 + 20) + 40):
                    if total_score <= 10:
                        print("Knock berhasil")
                        knock_settle(screen, total_score, bot_total, screen_width, screen_height)
                    else:
                        print("knock terlalu tinggi")
                elif mouse_pressed[0] and (screen_width - 140 < mouse_x < screen_width - 20) and (screen_height // 2 + 4 * (40 + 20) < mouse_y < screen_height // 2 + 4 * (40 + 20) + 40):
                    print("Load Bot Card button clicked!")
                    if os.path.exists(DETECTED_CARDS_FILE):
                        with open(DETECTED_CARDS_FILE, "r") as file:
                            detected_cards_file = [line.strip() for line in file.readlines()]
                        debug_print(f"Detected cards from file: {detected_cards_file}")
                        melds_set(detected_cards_file)
                        melds_run(detected_cards_file)
                        update_virtual_deck(detected_cards_file)
                    load_bot_cards()  
                    cards_loaded=True
                
        screen.blit(background_image, (0, 0))
        
        if show_cards and cards_loaded:
            display_cards(screen, detected_cards_file, card_images, screen_width, screen_height)
            set_meld(screen, set_list, card_images, screen_width, screen_height)
            set_run(screen, run_list, card_images, screen_width, screen_height)
            bot_set(screen, bot_set_list, card_images, screen_width, screen_height)
            bot_run(screen, bot_run_list, card_images, screen_width, screen_height)
 
        if show_cards and card_remove:
            discard_cards(screen, discarded_cards, card_images, screen_width, screen_height)
            
        draw_buttons(screen, screen_width, screen_height)
        display_bot_cards(screen, bot_detected_cards, card_images, screen_width, screen_height)
        
        total_score2 = calculate_score(detected_cards_file)
        font = pygame.font.Font(UIfont, 35) 
        text = f"Player Deadwood ({total_score2})"  
        text_surface = font.render(text, True, (255, 255, 255)) 
        screen.blit(text_surface, (1580, 1030))  
        
        screen.blit(decks, (150,( screen_height//2)-176))


        textbotcard = font.render(botcard, True, (255, 255, 255))   
        screen.blit(textbotcard, (20, 240))  
        
        
        bot_score = bot_calculate_score(bot_detected_cards)
        textbot = f"Bot Deadwood ({bot_score})"  
        textbot_surface = font.render(textbot, UIfont, (255, 255, 255)) 
        screen.blit(textbot_surface, (20, 20))  
        
        settings = "Tekan M untuk pembacaan P/B || Tekan S untuk mulai/stop scanner."  
        setfont = pygame.font.Font(UIfont, 18) 
        set_surface = setfont.render(settings, UIfont, (211, 211, 211)) 
        screen.blit(set_surface, (screen_width//2-320, screen_height//2+246))  
        
        text2 = "Melds"  
        text_surface2 = font.render(text2, True, (255, 255, 255)) 
        screen.blit(text_surface2, (20, screen_height-240)) 
        screen.blit(text_surface2, (screen_width- 110, 20))         
        
        decknum = f"Deck: {len(virtual_deck_list)}"  
        decknumsurf = font.render(decknum, True, (255, 255, 255)) 
        screen.blit(decknumsurf, (200,( screen_height//2)-220) ) 
        
        if cek_gin(total_score, detected_cards_file, screen, screen_width, screen_height):
            print("Player Gin!")
            
        if bot_total <= 10 and len(bot_detected_cards) > 0  and len(detected_cards_file) > 0:
            print()
            print("Bot knock settle!")
            knock_settle(screen, total_score, bot_total, screen_width, screen_height)
            
        if read_bot:
            pygame.draw.circle(screen, (0, 0, 255), (screen_width//2 -350 , screen_height//2+230), 20)  
            
            font = pygame.font.Font(None, 36)  
            text = font.render("B", True, (255, 255, 255))  
            
            text_rect = text.get_rect(center=(screen_width//2 -350 , screen_height//2+230))  
            
            screen.blit(text, text_rect)
            
        if not read_bot:
            pygame.draw.circle(screen, (255, 0, 0), (screen_width//2 -350 , screen_height//2+230), 20)  
            
            font = pygame.font.Font(None, 36)  
            text = font.render("P", True, (255, 255, 255))  
            
            text_rect = text.get_rect(center=(screen_width//2 -350 , screen_height//2+230)) 
            
            screen.blit(text, text_rect)
        
        #cv2.imshow("mask",mask)
        #cv2.imshow("Frame", frame)  # Menampilkan frame asli
        #cv2.waitKey(1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
        screen.blit(frame_surface, ((screen_width-640)//2, (screen_height-480)//2))
        pygame.display.flip()

        if cv2.waitKey(1) == ord('q'):
            cam.release()
            pygame.quit()
            break

def draw_buttons(screen, screen_width, screen_height):
    button_width, button_height = 180, 40
    margin = 20
    pygame.draw.rect(screen, (152, 251, 152), (screen_width - button_width - margin, screen_height // 2 - button_height - margin, button_width, button_height))
    font = pygame.font.Font(None, 36)
    draw_text = font.render("Draw", True, (0, 0, 0)) 
    screen.blit(draw_text, (screen_width - button_width - margin + (button_width - draw_text.get_width()) // 2, screen_height // 2 - button_height - margin + (button_height - draw_text.get_height()) // 2))

    pygame.draw.rect(screen, (255, 182, 193), (screen_width - button_width - margin, screen_height // 2 + margin, button_width, button_height))
    discard_text = font.render("Discard", True, (0, 0, 0)) 
    screen.blit(discard_text, (screen_width - button_width - margin + (button_width - discard_text.get_width()) // 2, screen_height // 2 + margin + (button_height - discard_text.get_height()) // 2))

    pygame.draw.rect(screen, (173, 216, 230), (screen_width - button_width - margin, screen_height // 2 + 2 * (button_height + margin), button_width, button_height))
    confirm_text = font.render("Konfirmasi", True, (0, 0, 0)) 
    screen.blit(confirm_text, (screen_width - button_width - margin + (button_width - confirm_text.get_width()) // 2, screen_height // 2 + 2 * (button_height + margin) + (button_height - confirm_text.get_height()) // 2))

    pygame.draw.rect(screen, (255, 224, 178), (screen_width - button_width - margin, screen_height // 2 + 3 * (button_height + margin), button_width, button_height))
    knock_text = font.render("Knock", True, (0, 0, 0)) 
    screen.blit(knock_text, (screen_width - button_width - margin + (button_width - knock_text.get_width()) // 2, screen_height // 2 + 3 * (button_height + margin) + (button_height - knock_text.get_height()) // 2))

    pygame.draw.rect(screen, (144, 238, 144), (screen_width - button_width - margin, screen_height // 2 + 4 * (button_height + margin), button_width, button_height))
    load_bot_card_text = font.render("Mulai", True, (0, 0, 0))  
    screen.blit(load_bot_card_text, (screen_width - button_width - margin + (button_width - load_bot_card_text.get_width()) // 2, screen_height // 2 + 4 * (button_height + margin) + (button_height - load_bot_card_text.get_height()) // 2))

    
def display_cards(screen, detected_cards, card_images, screen_width, screen_height):
    background_image = pygame.image.load("gincode.png")  
    background_image = pygame.transform.scale(background_image, (screen_width, screen_height))  

    screen.blit(background_image, (0, 0)) 

    num_cards = len(detected_cards)
    total_width = CARD_WIDTH * num_cards + (num_cards - 1) * CARD_SPACING
    x_offset_cards = (screen_width - total_width) // 2
    y_offset_cards = screen_height - CARD_HEIGHT - 10

    for card_name in detected_cards:
        if card_name in card_images:
            card_img = card_images[card_name]
            card_img_resized = pygame.transform.scale(card_img, (CARD_WIDTH, CARD_HEIGHT))
            screen.blit(card_img_resized, (x_offset_cards, y_offset_cards))
            x_offset_cards += CARD_WIDTH + CARD_SPACING

            
def display_bot_cards(screen, bot_detected_cards, card_images, screen_width, screen_height):
    num_cards = len(bot_detected_cards)
    total_width = CARD_WIDTH * num_cards + (num_cards - 1) * CARD_SPACING
    x_offset_cards = (screen_width - total_width) // 2  
    y_offset_cards = 10 

    for card_name in bot_detected_cards:
        if card_name in card_images:
            card_img = card_images[card_name]
            card_img_resized = pygame.transform.scale(card_img, (CARD_WIDTH, CARD_HEIGHT))
            screen.blit(card_img_resized, (x_offset_cards, y_offset_cards))
            x_offset_cards += CARD_WIDTH + CARD_SPACING 

def discard_cards(screen, discarded_cards, card_images, screen_width, screen_height):
    #screen.fill((0, 0, 0)) 
    x_offset_cards = (screen_width - CARD_WIDTH-300)
    y_offset_cards = (screen_height - CARD_HEIGHT) //2

    for card_name in discarded_cards:
        if card_name in card_images:
            card_img = card_images[card_name]
            card_img_resized = pygame.transform.scale(card_img, (CARD_WIDTH, CARD_HEIGHT))
            screen.blit(card_img_resized, (x_offset_cards, y_offset_cards))
            
def set_meld(screen, set_list, card_images, screen_width, screen_height):
    width_meld = CARD_WIDTH - 80
    height_meld = CARD_HEIGHT - 120
    x_offset_cards = 0
    y_offset_cards = (screen_height - height_meld)

    for card_name in set_list:
        if card_name in card_images:
            card_img = card_images[card_name]
            card_img_resized = pygame.transform.scale(card_img, (width_meld, height_meld))

            red_surface = pygame.Surface((width_meld, height_meld), pygame.SRCALPHA)
            red_surface.fill((255, 0, 0, 128))  

            screen.blit(card_img_resized, (x_offset_cards, y_offset_cards))
            screen.blit(red_surface, (x_offset_cards, y_offset_cards))  

            x_offset_cards += width_meld + 5

def bot_set(screen, bot_set_list, card_images, screen_width, screen_height):
    width_meld = CARD_WIDTH - 80
    height_meld = CARD_HEIGHT - 120
    x_offset_cards = 1920 - width_meld
    y_offset_cards = height_meld

    for card_name in bot_set_list:
        if card_name in card_images:
            card_img = card_images[card_name]
            card_img_resized = pygame.transform.scale(card_img, (width_meld, height_meld))

            red_surface = pygame.Surface((width_meld, height_meld), pygame.SRCALPHA)
            red_surface.fill((255, 0, 0, 128))  

            screen.blit(card_img_resized, (x_offset_cards, y_offset_cards))
            screen.blit(red_surface, (x_offset_cards, y_offset_cards))  

            x_offset_cards += -width_meld -5
            
def set_run(screen, run_list, card_images, screen_width, screen_height):
    width_meld = CARD_WIDTH-80
    height_meld = CARD_HEIGHT-120
    x_offset_cards = 0
    y_offset_cards = (screen_height-(2*height_meld+20))

    for card_name in run_list:
        if card_name in card_images:
            card_img = card_images[card_name]
            card_img_resized = pygame.transform.scale(card_img, (width_meld, height_meld))
            
            red_surface = pygame.Surface((width_meld, height_meld), pygame.SRCALPHA)
            red_surface.fill((0, 0, 255, 128)) 

            screen.blit(card_img_resized, (x_offset_cards, y_offset_cards))
            screen.blit(red_surface, (x_offset_cards, y_offset_cards))  
            
            x_offset_cards += width_meld + 5
            
def bot_run(screen, bot_run_list, card_images, screen_width, screen_height):
    width_meld = CARD_WIDTH-80
    height_meld = CARD_HEIGHT-120
    x_offset_cards = 1920 - width_meld
    y_offset_cards = (2*height_meld+2)

    for card_name in bot_run_list:
        if card_name in card_images:
            card_img = card_images[card_name]
            card_img_resized = pygame.transform.scale(card_img, (width_meld, height_meld))
            
            red_surface = pygame.Surface((width_meld, height_meld), pygame.SRCALPHA)
            red_surface.fill((0, 0, 255, 128)) 

            screen.blit(card_img_resized, (x_offset_cards, y_offset_cards))
            screen.blit(red_surface, (x_offset_cards, y_offset_cards))              
            
            x_offset_cards += -width_meld -5

if __name__ == "__main__":
    main()
