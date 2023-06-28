#include <iostream>
#include <vector>
#include <string>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>

SDL_Window *window = nullptr;
SDL_Renderer *renderer = nullptr;

struct Canvas {
    int x = 50;
    int y = 50;

    int w = 300;
    int h = 300;

    int ww = 28;
    int hh = 28;

    std::vector<std::vector<int>> data;
};

void InitCanvas(Canvas* canvas) {
    canvas->data.clear();
    for (int i = 0; i < 28; i++) {
        std::vector<int> column;
        for (int j = 0; j < 28; j++) {
            column.push_back(0);
        }
        canvas->data.push_back(column);
    }
}

int main(){

    SDL_Init(SDL_INIT_EVERYTHING);
    IMG_Init(IMG_INIT_JPG);
    TTF_Init();

    window = SDL_CreateWindow(
        "Computers and Technology",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        800, 500, 0);

    renderer = SDL_CreateRenderer(
        window,
        -1,
        SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_ACCELERATED);

    int mouseX = 0;
    int mouseY = 0;

    Canvas canvas;
    InitCanvas(&canvas);

    const char *backgroundImagePath = "res/background.jpeg";
    auto backgroundTexture = IMG_LoadTexture(renderer, backgroundImagePath);

    const SDL_Rect canvasRect = (SDL_Rect){canvas.x, canvas.y, canvas.w, canvas.h};

    bool mouseDown = false;

    std::cout << "Controls" << std::endl;
    std::cout << "Quit: Escape" << std::endl;
    std::cout << "Draw: Left click" << std::endl;
    std::cout << "Erase: Right click" << std::endl;
    std::cout << "Clear: C" << std::endl;


    bool run = true;
    while (run) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_WINDOWEVENT:
                if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
                    run = false;
                    break;
                }
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                case SDLK_ESCAPE:
                    run = false;
                    break;
                case SDLK_c:
                    InitCanvas(&canvas);
                    break;
                }
            case SDL_MOUSEBUTTONDOWN:
                mouseDown = true;
                break;
            case SDL_MOUSEBUTTONUP:
                mouseDown = false;
                break;
            default:
                break;
            }
        }

        // update mouse
        SDL_GetMouseState(&mouseX, &mouseY);
        SDL_Rect mouseRect = (SDL_Rect){mouseX, mouseY, 2, 2};

        // clear
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // background image
        SDL_RenderCopy(renderer, backgroundTexture, NULL, NULL);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderFillRect(renderer, &canvasRect);

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j  < 28; j++) {

                int pixelScreenSize = canvas.w / (canvas.ww - 1);
                int thisX = canvas.x + (i * pixelScreenSize);
                int thisY = canvas.y + (j * pixelScreenSize);
                SDL_Rect pixelRect = (SDL_Rect){thisX, thisY, pixelScreenSize, pixelScreenSize};

                if (canvas.data[i][j] == 1) {
                    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
                    SDL_RenderFillRect(renderer, &pixelRect);
                }

                // draw mouse position on grid
                if (SDL_HasIntersection(&mouseRect, &pixelRect)) {
                    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
                    SDL_RenderDrawRect(renderer, &pixelRect);
                    if (mouseDown) {
                        canvas.data[i][j] = 1;
                    }
                }
            }
        }
        
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderDrawRect(renderer, &canvasRect);

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    TTF_Quit();
    IMG_Quit();
    SDL_Quit();

    return 0;
}