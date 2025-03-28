#ifndef PHYSICSENGINE_TIMER_H
#define PHYSICSENGINE_TIMER_H

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>
#include <vector>
#include <cuda_runtime.h>

class PlsTimer {
public:
    // delete copy
    PlsTimer(const PlsTimer&) = delete;
    PlsTimer& operator=(const PlsTimer&) = delete;

    // get instance
    HOST_FUNC static PlsTimer& GetInstance() {
        static PlsTimer instance;
        return instance;
    }

    // host ticker start
    HOST_FUNC void HostStart(const std::string& tag) {
        hostTimers[tag].push_back(std::chrono::high_resolution_clock::now());
    }

    // host ticker end
    HOST_FUNC void HostStop(const std::string& tag) {
        auto end = std::chrono::high_resolution_clock::now();
        auto& starts = hostTimers[tag];
        if (!starts.empty()) {
            auto start = starts.back();
            starts.pop_back();
            hostDurations[tag].push_back(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6
            );
        }
    }

    // device time ticker start
    HOST_FUNC void DeviceStart(const std::string& tag) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cudaEvents[tag].push_back({ start, stop });
    }

    // device time ticker end
    HOST_FUNC void DeviceStop(const std::string& tag) {
        auto& events = cudaEvents[tag];
        if (!events.empty()) {
            auto [start, stop] = events.back();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            cudaDurations[tag].push_back(ms);

            events.pop_back();
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    // print all time information
    HOST_FUNC void PrintAll() {
        PLS_INFO("=== Host Timers ===");
        for (auto& [tag, times] : hostDurations) {
            double avg = 0;
            for (auto t : times) avg += t;
            avg /= times.size();
            PLS_INFO("[Host] {}: {} ms (count: {})", tag, avg, times.size());
        }

        PLS_INFO("=== Device Timers ===");
        for (auto& [tag, times] : cudaDurations) {
            double avg = 0;
            for (auto t : times) avg += t;
            avg /= times.size();
            PLS_INFO("[Device] {}: {} ms (count: {})", tag, avg, times.size());
        }
    }

    // clear all data
    HOST_FUNC void Clear() {
        hostTimers.clear();
        hostDurations.clear();
        cudaEvents.clear();
        cudaDurations.clear();
    }

private:
    // private construction
    HOST_FUNC PlsTimer() = default;

    // host timer data
    std::unordered_map<std::string, std::vector<std::chrono::high_resolution_clock::time_point>> hostTimers;
    std::unordered_map<std::string, std::vector<double>> hostDurations;

    // device timer data
    std::unordered_map<std::string, std::vector<std::pair<cudaEvent_t, cudaEvent_t>>> cudaEvents;
    std::unordered_map<std::string, std::vector<float>> cudaDurations;
};

#endif