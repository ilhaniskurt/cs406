TARGET_EXEC := run

BUILD_DIR := ./build
SRC_DIRS := ./src

SRCS := $(shell find $(SRC_DIRS) -name '*.cpp')
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INCLUDE := $(addprefix -I,$(INC_DIRS))

CXX = g++
CXXFLAGS = -std=c++11 -O3 -pedantic-errors -Wall -Wextra -fopenmp

# The final build step.
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)


# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -MP -o $@


.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

-include $(DEPS)
