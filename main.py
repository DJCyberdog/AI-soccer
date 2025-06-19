import pygame
import yaml
import os
import time
from env.soccer_env import SoccerEnv
from gui.overlay import Overlay
from utils.team_loader import load_team

CONFIG_PATH = "config/match_config.yaml"


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def init_game():
    pygame.init()
    config = load_config(CONFIG_PATH)
    screen = pygame.display.set_mode((config['field']['width'], config['field']['height']))
    pygame.display.set_caption("2D Soccer RL")
    clock = pygame.time.Clock()
    return screen, clock, config


def run_game():
    screen, clock, config = init_game()
    env = SoccerEnv(config)

    # Load teams
    team_red = load_team(config['teams']['red'], "red", env)
    team_blue = load_team(config['teams']['blue'], "blue", env)
    env.set_teams(team_red, team_blue)

    overlay = Overlay(config)
    running = True

    while running:
        screen.fill((0, 128, 0))  # Grass green background

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Step environment
        env.step()

        # Draw everything
        env.render(screen)
        overlay.draw(screen, env)

        pygame.display.flip()
        clock.tick(config['game']['fps'])

    # Save Q-tables on exit
    env.save_models()
    pygame.quit()


if __name__ == "__main__":
    run_game()
