#ifndef HPIPE_UTILS_HPP
#define HPIPE_UTILS_HPP

#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>

/**
 * H-Pipe Logging Utilities
 */

// Get current timestamp formatted as [HH:MM:SS.mmm]
inline std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c); 

    std::ostringstream oss;
    oss << "[" << std::put_time(&now_tm, "%H:%M:%S") 
        << "." << std::setfill('0') << std::setw(3) << ms.count() << "]";
    return oss.str();
}

// LOG macro for consistent timestamped output
#define LOG(msg) std::cout << getTimestamp() << " " << msg << std::endl

#endif // HPIPE_UTILS_HPP
