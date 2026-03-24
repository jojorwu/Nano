#include <cstdint>
#include <iostream>

extern "C" {
    /**
     * Example plugin function that implements a periodic burst of activity.
     * bus_ptr: pointer to the somatic potentials buffer.
     * bus_size: number of neurons.
     * tick: current simulation tick.
     */
    void my_plugin_tick(int32_t* bus_ptr, uint32_t bus_size, uint32_t tick) {
        if (tick % 10 == 0) {
            // Inject a burst of potential into the first 10 neurons
            for (uint32_t i = 0; i < 10 && i < bus_size; ++i) {
                bus_ptr[i] += 1000;
            }
        }
    }
}
