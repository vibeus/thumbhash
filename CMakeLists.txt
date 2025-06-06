cmake_minimum_required(VERSION 3.15)
project(libthumbhash
    VERSION 0.1.0
    LANGUAGES C
    DESCRIPTION "A C implementation of ThumbHash image placeholder generation"
)


# Library sources and headers
set(LIBTHUMBHASH_SOURCES
    src/thumbhash.c
)

set(LIBTHUMBHASH_HEADERS
    include/thumbhash/thumbhash.h
)

# Create the library target
add_library(thumbhash ${LIBTHUMBHASH_SOURCES} ${LIBTHUMBHASH_HEADERS})

target_sources(thumbhash
    PUBLIC
        ${LIBTHUMBHASH_HEADERS}
)

set_target_properties(thumbhash PROPERTIES
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF  # For strict C99 compliance
)

# Public include directory (consumers will automatically get this)
target_include_directories(thumbhash
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)

target_compile_options(thumbhash PUBLIC
    $<$<CONFIG:Debug>:-Wall -fstrict-aliasing -O0 -g -DDEBUG>
    $<$<CONFIG:Release>:-Wall -fstrict-aliasing -flto -O3 -DNDEBUG>
)

target_link_options(thumbhash PUBLIC "-lm")

