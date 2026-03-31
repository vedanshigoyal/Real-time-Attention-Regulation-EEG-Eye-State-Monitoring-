# game.py (Modification 4: Silent Version - No Assets)

import pygame
import mne
import os
import numpy as np
from scipy.signal import hilbert
import joblib
import time
import random
import warnings

# Suppress MNE warnings for a cleaner console output
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')

# --- Advanced Feature Extraction Function (no changes) ---
def extract_plv_features_manual(epochs):
    data = epochs.get_data(picks='eeg')
    sfreq = epochs.info['sfreq']
    n_epochs, n_channels, _ = data.shape
    data_alpha = mne.filter.filter_data(data, sfreq, 8, 13, verbose=False)
    analytic_signal = hilbert(data_alpha, axis=2)
    phase_data = np.angle(analytic_signal)
    plv_features = []
    for epoch_idx in range(n_epochs):
        plv_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = phase_data[epoch_idx, i, :] - phase_data[epoch_idx, j, :]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv
        triu_indices = np.triu_indices(n_channels, k=1)
        plv_features.append(plv_matrix[triu_indices])
    return np.array(plv_features)

# --- High Score Functions (no changes) ---
def load_high_score():
    try:
        with open("highscore.txt", "r") as f:
            return int(f.read())
    except (IOError, ValueError):
        return 0

def save_high_score(new_score, current_high):
    if new_score > current_high:
        with open("highscore.txt", "w") as f:
            f.write(str(new_score))
        return new_score
    return current_high

# --- Game Classes (no changes) ---
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((50, 60), pygame.SRCALPHA)
        self.original_image = self.image
        self.draw_character(self.image, (255, 50, 50))
        self.rect = self.image.get_rect(midbottom = (100, 450))
        self.gravity = 0
        self.is_jumping = False

    def draw_character(self, surface, color):
        surface.fill((0,0,0,0))
        pygame.draw.rect(surface, color, (0, 10, 50, 50), border_radius=10)
        pygame.draw.circle(surface, 'white', (25, 30), 10)
        pygame.draw.circle(surface, 'black', (28, 32), 5)

    def apply_gravity(self):
        self.gravity += 1
        self.rect.y += self.gravity
        if self.rect.bottom >= 450:
            self.rect.bottom = 450
            if self.is_jumping:
                self.is_jumping = False
                self.image = pygame.transform.scale(self.original_image, (55, 55))

    def jump(self, focus_fuel):
        if self.rect.bottom == 450 and focus_fuel >= 35:
            self.gravity = -22
            self.is_jumping = True
            self.image = pygame.transform.scale(self.original_image, (45, 65))
            return True
        return False

    def update_color(self, focus_fuel, mental_state):
        if focus_fuel >= 35:
            color = (50, 255, 50)
        else:
            color = (255, 50, 50)
        self.draw_character(self.original_image, color)
        if mental_state == "REST":
            aura_surf = pygame.Surface((60, 70), pygame.SRCALPHA)
            pygame.draw.circle(aura_surf, (0, 150, 255, 90), (30, 35), 30)
            self.image.blit(self.original_image, (5,5))
            self.image.blit(aura_surf, (0,0), special_flags=pygame.BLEND_RGBA_ADD)

    def update(self):
        self.apply_gravity()

class Obstacle(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        width = random.randint(40, 60)
        height = random.randint(80, 150)
        self.image = pygame.Surface((width, height)).convert_alpha()
        self.image.fill((0,0,0,0))
        points = [(width/2, 0), (width, height/3), (width, 2*height/3), (width/2, height), (0, 2*height/3), (0, height/3)]
        pygame.draw.polygon(self.image, (180, 220, 255), points)
        pygame.draw.polygon(self.image, 'white', points, 2)
        self.rect = self.image.get_rect(midbottom = (random.randint(900, 1100), 450))

    def update(self):
        self.rect.x -= 7
        if self.rect.right < 0: self.kill()

class Particle(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        self.image = pygame.Surface((8, 8))
        pygame.draw.circle(self.image, 'white', (4, 4), 4)
        self.rect = self.image.get_rect(center=pos)
        self.velocity = pygame.math.Vector2(random.uniform(-2, 2), random.uniform(-4, -1))
        self.lifetime = 20

    def update(self):
        self.rect.move_ip(self.velocity)
        self.lifetime -= 1
        if self.lifetime <= 0: self.kill()

# --- Data Loading Function (no changes) ---
def load_and_preprocess_eeg(subject_number):
    data_folder = "eeg-during-mental-arithmetic-tasks-1.0.0"
    subject_id = f"Subject{subject_number:02d}"
    low_attention_file = os.path.join(data_folder, f"{subject_id}_1.edf")
    high_attention_file = os.path.join(data_folder, f"{subject_id}_2.edf")

    if not os.path.exists(low_attention_file) or not os.path.exists(high_attention_file):
        print(f"Error: Data for Subject {subject_number} not found.")
        return None, None

    def process_file(file_path):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.pick('eeg', exclude='bads')
        raw.filter(1., 40., fir_design='firwin', verbose=False)
        epochs = mne.make_fixed_length_epochs(raw, duration=2, overlap=1, preload=True, verbose=False)
        return epochs

    low_epochs = process_file(low_attention_file)
    high_epochs = process_file(high_attention_file)
    n_low, n_high = len(low_epochs), len(high_epochs)
    n_balanced = min(n_low, n_high)
    low_epochs_balanced = low_epochs[np.random.choice(n_low, n_balanced, replace=False)]
    high_epochs_balanced = high_epochs[np.random.choice(n_high, n_balanced, replace=False)]
    return low_epochs_balanced, high_epochs_balanced

# --- 1. SETUP ---
pygame.init()
WIDTH, HEIGHT = 800, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Focus Jumper")
font = pygame.font.Font(None, 50)
small_font = pygame.font.Font(None, 30)
clock = pygame.time.Clock()
obstacle_timer = pygame.USEREVENT + 1
game_state = "SELECTION"

player_buttons = []
for i in range(1, 6):
    button_rect = pygame.Rect(WIDTH/2 - 100, 100 + (i-1)*60, 200, 50)
    player_buttons.append({'rect': button_rect, 'id': i})

# --- Load BCI Model, Scaler, and High Score ---
try:
    model = joblib.load('connectivity_model.joblib')
    scaler = joblib.load('connectivity_scaler.joblib')
except FileNotFoundError:
    print("ERROR: 'connectivity_model.joblib' or 'connectivity_scaler.joblib' not found.")
    pygame.quit()
    exit()
high_score = load_high_score()

# Background
background_surf = pygame.Surface((WIDTH, HEIGHT)); background_surf.fill((10, 10, 30))
stars = [{'rect': pygame.Rect(random.randint(0, WIDTH), random.randint(0, HEIGHT), random.randint(1, 3), random.randint(1, 3)), 'speed': random.randint(1, 3)} for _ in range(150)]

# --- 2. MAIN GAME LOOP ---
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: pygame.quit(); exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: pygame.quit(); exit()

        if game_state == "SELECTION":
            if event.type == pygame.MOUSEBUTTONDOWN:
                for button in player_buttons:
                    if button['rect'].collidepoint(event.pos):
                        selected_player = button['id']
                        game_state = "LOADING"

        elif game_state == "GAME_ACTIVE":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if player.sprite.jump(focus_fuel):
                        focus_fuel -= 35
                        for _ in range(10): particles.add(Particle(player.sprite.rect.midbottom))
                if event.key == pygame.K_m:
                    mental_state = "REST" if mental_state == "FOCUS" else "FOCUS"

            if event.type == obstacle_timer: obstacles.add(Obstacle())

        elif game_state == "GAME_OVER":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                game_state = "SELECTION"

    # --- State Machine Logic ---
    
    if game_state == "SELECTION":
        screen.blit(background_surf, (0,0))
        title_surf = font.render("Select a Player", True, "white")
        screen.blit(title_surf, (WIDTH/2 - 120, 30))
        high_score_surf = small_font.render(f"High Score: {high_score}", True, "yellow")
        screen.blit(high_score_surf, (WIDTH/2 - 60, 420))
        for button in player_buttons:
            pygame.draw.rect(screen, 'grey', button['rect'], border_radius=10)
            player_text = font.render(f"Player {button['id']}", True, "black")
            screen.blit(player_text, (button['rect'].x + 40, button['rect'].y + 10))

    elif game_state == "LOADING":
        screen.fill((10, 10, 30))
        loading_text = font.render(f"Loading & Processing Player {selected_player}'s Brainwaves...", True, "white")
        screen.blit(loading_text, (WIDTH // 2 - 380, HEIGHT // 2 - 30))
        pygame.display.flip()
        low_epochs, high_epochs = load_and_preprocess_eeg(selected_player)
        if low_epochs is None:
            game_state = "SELECTION"
        else:
            low_attention_features = extract_plv_features_manual(low_epochs)
            high_attention_features = extract_plv_features_manual(high_epochs)
            low_features_scaled = scaler.transform(low_attention_features)
            high_features_scaled = scaler.transform(high_attention_features)
            score = 0
            focus_fuel = 20
            mental_state = "FOCUS"
            player = pygame.sprite.GroupSingle(); player.add(Player())
            obstacles = pygame.sprite.Group()
            particles = pygame.sprite.Group()
            obstacle_timer_started = False
            low_feature_pointer = 0
            high_feature_pointer = 0
            last_update_time = time.time()
            game_state = "GAME_ACTIVE"

    elif game_state == "GAME_ACTIVE":
        # BCI Simulator Logic
        if time.time() - last_update_time > 1:
            last_update_time = time.time()
            if mental_state == "FOCUS":
                current_feature = high_features_scaled[high_feature_pointer].reshape(1, -1)
                prediction = model.predict(current_feature)[0]
                if prediction == 1:
                    focus_fuel += 70
                high_feature_pointer = (high_feature_pointer + 1) % len(high_features_scaled)
            else:
                current_feature = low_features_scaled[low_feature_pointer].reshape(1, -1)
                model.predict(current_feature)
                low_feature_pointer = (low_feature_pointer + 1) % len(low_features_scaled)
        
        fuel_decay = 0.4 if mental_state == "REST" else 0.1
        focus_fuel -= fuel_decay
        if focus_fuel < 0: focus_fuel = 0
        if focus_fuel > 100: focus_fuel = 100

        # Drawing
        screen.blit(background_surf, (0,0))
        for star in stars:
            star['rect'].x -= star['speed']; 
            if star['rect'].right < 0: star['rect'].left = WIDTH
            pygame.draw.rect(screen, 'white', star['rect'])
        pygame.draw.rect(screen, (40, 40, 60), (0, 450, WIDTH, 50))
        player.sprite.update_color(focus_fuel, mental_state)
        if not obstacle_timer_started and score > 300:
            pygame.time.set_timer(obstacle_timer, 1500)
            obstacle_timer_started = True

        player.draw(screen); player.update(); obstacles.draw(screen); obstacles.update()
        particles.draw(screen); particles.update()
        
        if pygame.sprite.spritecollide(player.sprite, obstacles, False):
            high_score = save_high_score(score // 100, high_score)
            game_state = "GAME_OVER"
            
        score += 1
        
        # UI Display
        display_score = score // 100
        score_surf = font.render(f'Score: {display_score}', True, 'white')
        screen.blit(score_surf, (10, 10))
        pygame.draw.rect(screen, (0,0,0,150), (WIDTH - 255, 5, 240, 40))
        pygame.draw.rect(screen, (255, 255, 0), (WIDTH - 255, 5, focus_fuel * 2.4, 40))
        pygame.draw.rect(screen, 'white', (WIDTH - 255, 5, 240, 40), 3, border_radius=5)
        focus_text = font.render(f'Focus', True, 'white')
        screen.blit(focus_text, (WIDTH - 350, 10))
        state_color = "cyan" if mental_state == "FOCUS" else "orange"
        state_surf = small_font.render(f'MODE: {mental_state} (M to switch)', True, state_color)
        screen.blit(state_surf, (10, 50))

    elif game_state == "GAME_OVER":
        final_score = score // 100
        screen.fill((10, 10, 30))
        game_over_surf = font.render('Game Over', True, 'red')
        screen.blit(game_over_surf, (WIDTH // 2 - 100, HEIGHT // 2 - 100))
        score_text_surf = small_font.render(f'Your Score: {final_score}', True, 'white')
        screen.blit(score_text_surf, (WIDTH // 2 - 80, HEIGHT // 2 - 40))
        high_score_text_surf = small_font.render(f'High Score: {high_score}', True, 'yellow')
        screen.blit(high_score_text_surf, (WIDTH // 2 - 80, HEIGHT // 2 - 10))
        restart_surf = small_font.render('Press Space to return to Player Selection', True, 'white')
        screen.blit(restart_surf, (WIDTH // 2 - 200, HEIGHT // 2 + 50))
    
    pygame.display.flip()
    clock.tick(60)
