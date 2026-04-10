#ifndef LAYER_H
#define LAYER_H

typedef struct {
    int    n_inputs;     /* number of entries in this layer */
    int    n_outputs;    /* number of neurons (outputs) */

    float *weights;     /* matrix [n_outputs × n_inputs] */
    float *biases;       /* vector [n_outputs]                     */
    float *activations; /* vector [n_outputs] - neuron output. */
    float *deltas;       /* vector [n_outputs] - used in backprop */

} Layer;

/* Allocates and initializes a layer with random weights */
Layer *layer_create(int n_inputs, int n_outputs);

/* Frees all memory from the layer */
void   layer_free(Layer *layer);

/* Calculates the activations given an input vector */
void   layer_forward(Layer *layer, float *input);

#endif /* LAYER_H */