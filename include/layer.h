#ifndef LAYER_H
#define LAYER_H

typedef struct {
    int    n_inputs;     /* número de entradas desta camada        */
    int    n_outputs;    /* número de neurónios (saídas)           */

    float *weights;      /* matriz [n_outputs × n_inputs]          */
    float *biases;       /* vector [n_outputs]                     */
    float *activations;  /* vector [n_outputs] - saída dos neuron. */
    float *deltas;       /* vector [n_outputs] - usado no backprop */

} Layer;

/* Aloca e inicializa uma camada com pesos aleatórios */
Layer *layer_create(int n_inputs, int n_outputs);

/* Liberta toda a memória da camada */
void   layer_free(Layer *layer);

/* Calcula as activações dado um vector de input */
void   layer_forward(Layer *layer, float *input);

#endif /* LAYER_H */