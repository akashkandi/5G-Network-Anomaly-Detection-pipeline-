/**
 * 5G Network Telemetry Preprocessor — C++ Implementation
 * ========================================================
 * Reads a raw CSV telemetry file, computes rolling mean and standard
 * deviation for each numeric metric column, outputs a preprocessed CSV,
 * and reports wall-clock timing for comparison against Python/pandas.
 *
 * Build:
 *   g++ -O2 -std=c++17 -o cpp/preprocessing cpp/preprocessing.cpp
 *
 * Usage:
 *   ./cpp/preprocessing <input.csv> <output.csv> [window_size]
 *
 * Example:
 *   ./cpp/preprocessing outputs/telemetry_raw.csv outputs/telemetry_cpp.csv 15
 *
 * Columns expected in input CSV (any order; others are passed through):
 *   latency_ms, packet_loss_pct, throughput_mbps,
 *   handover_count, signal_strength_dbm, cpu_util_pct
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ─── CSV Parser ──────────────────────────────────────────────────────────────

/**
 * Split a CSV line respecting double-quoted fields.
 */
std::vector<std::string> split_csv(const std::string &line) {
    std::vector<std::string> fields;
    bool in_quotes = false;
    std::string field;
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);
    return fields;
}

// ─── Rolling Statistics ───────────────────────────────────────────────────────

/**
 * Maintains an online rolling window for mean and standard deviation
 * using Welford's incremental algorithm for numerical stability.
 */
class RollingStats {
public:
    explicit RollingStats(int window) : window_(window) {}

    void push(double value) {
        window_data_.push_back(value);
        if (static_cast<int>(window_data_.size()) > window_) {
            window_data_.pop_front();
        }
    }

    double mean() const {
        if (window_data_.empty()) return 0.0;
        double s = std::accumulate(window_data_.begin(), window_data_.end(), 0.0);
        return s / static_cast<double>(window_data_.size());
    }

    double std_dev() const {
        int n = static_cast<int>(window_data_.size());
        if (n < 2) return 0.0;
        double m = mean();
        double sq_sum = 0.0;
        for (double v : window_data_) {
            sq_sum += (v - m) * (v - m);
        }
        return std::sqrt(sq_sum / static_cast<double>(n - 1));
    }

private:
    int window_;
    std::deque<double> window_data_;
};

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.csv> <output.csv> [window_size=15]\n";
        return 1;
    }

    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];
    const int window_size = (argc >= 4) ? std::stoi(argv[3]) : 15;

    const std::vector<std::string> TARGET_COLS = {
        "latency_ms", "packet_loss_pct", "throughput_mbps",
        "handover_count", "signal_strength_dbm", "cpu_util_pct"
    };

    std::cout << "[C++] Input:    " << input_path  << "\n";
    std::cout << "[C++] Output:   " << output_path << "\n";
    std::cout << "[C++] Window:   " << window_size  << "\n";

    // ── Open input ────────────────────────────────────────────────────────────
    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        std::cerr << "[C++] ERROR: Cannot open input file: " << input_path << "\n";
        return 1;
    }

    // ── Parse header ──────────────────────────────────────────────────────────
    std::string header_line;
    std::getline(infile, header_line);
    std::vector<std::string> headers = split_csv(header_line);

    // Find column indices for target metrics
    std::unordered_map<std::string, int> col_idx;
    for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
        // Trim whitespace
        std::string h = headers[i];
        h.erase(h.begin(), std::find_if(h.begin(), h.end(),
                [](unsigned char c) { return !std::isspace(c); }));
        h.erase(std::find_if(h.rbegin(), h.rend(),
                [](unsigned char c) { return !std::isspace(c); }).base(), h.end());
        col_idx[h] = i;
        headers[i] = h;
    }

    for (const auto &col : TARGET_COLS) {
        if (col_idx.find(col) == col_idx.end()) {
            std::cerr << "[C++] WARNING: Column '" << col << "' not found in header.\n";
        }
    }

    // ── Prepare output header ─────────────────────────────────────────────────
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "[C++] ERROR: Cannot open output file: " << output_path << "\n";
        return 1;
    }

    // Write output header: original columns + rolling stats columns
    for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
        outfile << headers[i];
        if (i + 1 < static_cast<int>(headers.size())) outfile << ",";
    }
    for (const auto &col : TARGET_COLS) {
        if (col_idx.count(col)) {
            outfile << "," << col << "_roll_mean_" << window_size;
            outfile << "," << col << "_roll_std_"  << window_size;
        }
    }
    outfile << "\n";

    // ── Initialise rolling statistics ─────────────────────────────────────────
    std::unordered_map<std::string, RollingStats> stats_map;
    for (const auto &col : TARGET_COLS) {
        if (col_idx.count(col)) {
            stats_map.emplace(col, RollingStats(window_size));
        }
    }

    // ── Process rows ──────────────────────────────────────────────────────────
    auto t_start = std::chrono::high_resolution_clock::now();

    long long rows_processed = 0;
    std::string row_line;

    while (std::getline(infile, row_line)) {
        if (row_line.empty()) continue;
        std::vector<std::string> fields = split_csv(row_line);

        // Ensure we have enough fields
        while (static_cast<int>(fields.size()) < static_cast<int>(headers.size())) {
            fields.push_back("");
        }

        // Pass through original fields
        for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
            outfile << fields[i];
            if (i + 1 < static_cast<int>(headers.size())) outfile << ",";
        }

        // Compute and write rolling stats
        for (const auto &col : TARGET_COLS) {
            auto it = col_idx.find(col);
            if (it == col_idx.end()) continue;

            int idx = it->second;
            double value = 0.0;
            bool is_missing = false;

            if (idx < static_cast<int>(fields.size()) && !fields[idx].empty()) {
                try {
                    value = std::stod(fields[idx]);
                } catch (...) {
                    is_missing = true;
                }
            } else {
                is_missing = true;
            }

            auto &rs = stats_map.at(col);
            if (!is_missing) {
                rs.push(value);
            }

            outfile << "," << rs.mean();
            outfile << "," << rs.std_dev();
        }

        outfile << "\n";
        ++rows_processed;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    infile.close();
    outfile.close();

    // ── Benchmark output ──────────────────────────────────────────────────────
    std::cout << "[C++] Rows processed : " << rows_processed        << "\n";
    std::cout << "[C++] Elapsed time   : " << elapsed_ms             << " ms\n";
    std::cout << "[C++] Throughput     : "
              << static_cast<long long>(rows_processed / (elapsed_ms / 1000.0))
              << " rows/sec\n";
    std::cout << "[C++] Output written : " << output_path             << "\n";

    return 0;
}
