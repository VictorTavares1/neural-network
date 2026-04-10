#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

typedef struct {
    int    n_layers;  /* total number of layers */
    Layer **layers;   /* array of pointers to each Layer */
} Network;

/* Creates a network with n layers.

sizes[] defines the number of neurons in each layer.

Example: sizes = {2, 4, 1} → 2 inputs, 4 neurons, 1 output */
Network *network_create(int n_layers, int *sizes);

/* Frees up all network memory */
void     network_free(Network *net);

/* Executes the forward pass on all layers */
void     network_forward(Network *net, float *input);

/* Trains the network with an input, target pair */
void     network_train(Network *net, float *input,
                       float *target, float learning_rate);

#endif /* NETWORK_H */