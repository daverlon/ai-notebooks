#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>


#include <SDL2/SDL.h>

// #define showdebug

void predict();

SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

int mouseX;
int mouseY;
bool mouseDown;

std::vector<std::vector<float>> weights;

std::vector<std::vector<float>> canvasData(28, std::vector<float>(28)); // 0.0f - 1.0f

std::vector<float> predictions(10);

std::vector<std::vector<float>> extractWeights(const std::string& filename) {
    std::ifstream inputFile(filename);
    std::vector<std::vector<float>> weights;

    if (inputFile.is_open()) {
        std::string line;

        while (std::getline(inputFile, line)) {
            std::istringstream iss(line);
            std::vector<float> weightRow;
            float weight;

            while (iss >> weight) {
                weightRow.push_back(weight);
            }

            if (!weightRow.empty()) {
                weights.push_back(weightRow);
            }
        }

        inputFile.close();
    } else {
#ifdef showdebug
        std::cerr << "Unable to open file: " << filename << std::endl;
#endif
    }

    return weights;
}

std::vector<float> flatten(const std::vector<std::vector<float>>& input) {
    std::vector<float> output;

    for (const auto& row : input) {
        output.insert(output.end(), row.begin(), row.end());
    }

#ifdef showdebug
    std::cout << "Flatten -> (" << output.size() << ",)" << std::endl;
#endif
    return output;
}

std::vector<std::vector<float>> reshape(const std::vector<float>& input, int rows, int cols) {
    std::vector<std::vector<float>> output(rows, std::vector<float>(cols));

    if (input.size() != rows * cols) {
        std::cerr << "Invalid reshape dimensions." << std::endl;
        return output;
    }

    int index = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[i][j] = input[index];
            index++;
        }
    }

    return output;
}

std::vector<std::vector<float>> expandDims(const std::vector<float>& x, int axis) {
    std::vector<std::vector<float>> expanded;
    
    if (axis < 0 || axis > static_cast<int>(x.size())) {
        std::cerr << "Invalid axis value." << std::endl;
        return expanded;
    }

    expanded.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        if (i == static_cast<size_t>(axis)) {
            expanded.push_back({x[i]});
        } else {
            expanded.push_back({x[i]});
        }
    }

    return expanded;
}

std::vector<float> Linear(const std::vector<float>& x, const std::vector<std::vector<float>>& w, const std::vector<float>& bias) {
    if (x.size() != w[0].size()) {
        std::cerr << "Input vector size mismatch." << std::endl;
        return {};
    }

    std::vector<float> result(w.size(), 0.0f);
    for (size_t i = 0; i < w.size(); i++) {
        for (size_t j = 0; j < x.size(); j++) {
            result[i] += x[j] * w[i][j];
        }
        result[i] += bias[i];
    }

#ifdef showdebug
    std::cout << "Linear -> (" << result.size() << ",)" << std::endl;
#endif
    return result;
}

std::vector<float> ReLU(const std::vector<float>& input) {
    std::vector<float> result;
    result.reserve(input.size());

    for (float value : input) {
        result.push_back(std::max(0.0f, value));
    }

#ifdef showdebug
    std::cout << "ReLU -> (" << result.size() << ",)" << std::endl;
#endif
    return result;
}

std::vector<float> LogSoftmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    float maxVal = input[0];
    for (float value : input) {
        maxVal = std::max(maxVal, value);
    }

    float sumExp = 0.0f;
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::exp(input[i] - maxVal);
        sumExp += output[i];
    }

    for (size_t i = 0; i < output.size(); i++) {
        output[i] = std::log(output[i] / sumExp);
    }

    return output;
}

void initCanvasData() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            canvasData[y][x] = dis(gen);  // Assign random float value
        }
    }
}

void clearCanvas() {
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            canvasData[y][x] = 0.0f;  // Reset pixel values to 0.0
        }
    }
}

void clean() {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
}

void renderCanvas() {
    SDL_Rect canvasRect = SDL_Rect{50, 50, 300, 300};

    // canvas background
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderFillRect(renderer, &canvasRect);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {

            const float thisPixel = canvasData[y][x];
            const float scaled = thisPixel * 255;
        
            SDL_SetRenderDrawColor(renderer, scaled, scaled, scaled, 255);

            SDL_FRect pixelRect = SDL_FRect{
                canvasRect.x + x * (300.0f / 28.0f),
                canvasRect.y + y * (300.0f / 28.0f),
                (300.0f / 28.0f),
                (300.0f / 28.0f)
            };

            SDL_RenderFillRectF(renderer, &pixelRect);
        }
    }
}

void renderResults() {

    const float topOffset = 45;

    SDL_FRect area = SDL_FRect{400.0f, 50.0f + topOffset, 350.0f, 300.0f - topOffset};

    // SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    // SDL_RenderFillRectF(renderer, &area);

    const float barGap = 5.0f; // gap between each bar

    const float barW = (area.w / 9) - (area.w / (barGap * 9));
    
    for (int i = 0; i < 10; i++) {
        SDL_FRect barRect = SDL_FRect{
            area.x + i * barW,
            area.y, 
            barW - barGap,
            area.h
        };

        if (i > 0) barRect.x += barGap * i;


        float pred = predictions[i]; // probability 0.0 - 1.0
        float yOffset = pred * 100.0f;

        barRect.y -= yOffset / area.h * 100.0f;
        barRect.h += yOffset / area.h * 100.0f;

        barRect.h = std::max(0.0f, barRect.h);

        barRect.y = std::min(barRect.y, barRect.y + barRect.h);


        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
        SDL_RenderFillRectF(renderer, &barRect);

        barRect = SDL_FRect{
            area.x + i * barW,
            area.y, 
            barW - barGap,
            area.h
        };
        if (i > 0) barRect.x += barGap * i;

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderDrawRectF(renderer, &barRect);
    }
}

int main() {

    window = SDL_CreateWindow(
        "Digit Classifier", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        800, 400, SDL_WINDOW_RESIZABLE
    );

    renderer = SDL_CreateRenderer(
        window,
        -1,
        SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_ACCELERATED
    );

    // initCanvasData();

    std::string filename = "./res/model_weights.txt";
    weights = extractWeights(filename);

#ifdef showdebug
    for (const auto& weightRow : weights) {
        for (float weight : weightRow) {
            std::cout << weight << " ";
        }
        std::cout << std::endl;
    }
#endif

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
                break;

            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        run = false;
                        break;
                    case SDLK_c:
                        clearCanvas();
                        break;
                    case SDLK_r:
                        initCanvasData();
                        break;
                    default:
                        break;
                }
                break;

            case SDL_MOUSEBUTTONDOWN:
                switch (event.button.button) {
                case SDL_BUTTON_LEFT:
                    mouseDown = true;
                    break;
                default:
                    break;
                }
                break;

            case SDL_MOUSEBUTTONUP:
                switch (event.button.button) {
                case SDL_BUTTON_LEFT:
                    mouseDown = false;
                default:
                    break;
                }
                break;

            case SDL_MOUSEMOTION:
                if (mouseDown) {

                    int canvasX = (mouseX - 50) / (300 / 28);
                    int canvasY = (mouseY - 50) / (300 / 28);

                    if (canvasX >= 0 && canvasX < 28 && canvasY >= 0 && canvasY < 28) {
                        canvasData[canvasY][canvasX] = 1.0f;
                        predict();
                    }
                }
                break;

            default:
                break;
            }
        }

        SDL_GetMouseState(&mouseX, &mouseY);



        SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
        SDL_RenderClear(renderer);

        renderCanvas();
        renderResults();


        SDL_RenderPresent(renderer);
    }

    clean();

    return 0;
}

void predict() {

    auto x = canvasData;
#ifdef showdebug
    std::cout << std::endl << "Input -> (" << x.size() << "," << x[0].size() << ")" << std::endl;
#endif

    auto flat = flatten(x);

    auto pred = Linear(flat, reshape(weights[0], 64, 784), weights[1]);
    pred = ReLU(pred);
    pred = Linear(pred, reshape(weights[2], 32, 64), weights[3]);
    pred = ReLU(pred);
    pred = Linear(pred, reshape(weights[4], 10, 32), weights[5]);

    // no need to expand here
    
    pred = LogSoftmax(pred);

#ifdef showdebug
    for (int i = 0; i < pred.size(); i++)
        std::cout << std::exp(pred[i])*100.0f << ",";
    std::cout << std::endl;
#endif

    predictions = pred;
}