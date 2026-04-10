#include <stdlib.h>
#include <math.h>
#include "layer.h"

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

Layer *layer_create(int n_inputs, int n_outputs) {
    Layer *layer = malloc(sizeof(Layer));

    layer->n_inputs  = n_inputs;
    layer->n_outputs = n_outputs;

    layer->weights     = malloc(sizeof(float) * n_inputs * n_outputs);
    layer->biases      = malloc(sizeof(float) * n_outputs);
    layer->activations = malloc(sizeof(float) * n_outputs);
    layer->deltas      = malloc(sizeof(float) * n_outputs);

    for (int i = 0; i < n_inputs * n_outputs; i++)
        layer->weights[i] = ((float)rand() / RAND_MAX) - 0.5f;

    for (int i = 0; i < n_outputs; i++)
        layer->biases[i] = 0.0f;

    return layer;
}

void layer_free(Layer *layer) {
    free(layer->weights);
    free(layer->biases);
    free(layer->activations);
    free(layer->deltas);
    free(layer);
}

void layer_forward(Layer *layer, float *input) {
    for (int i = 0; i < layer->n_outputs; i++) {
        float soma = layer->biases[i];

        for (int j = 0; j < layer->n_inputs; j++)
            soma += input[j] * layer->weights[i * layer->n_inputs + j];

        layer->activations[i] = sigmoid(soma);
    }
}