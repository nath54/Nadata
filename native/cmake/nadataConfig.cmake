# nadataConfig.cmake — consumer entry point for a staged Nadata
# distribution.
#
# The staged layout (tarball) keeps everything in one directory:
#
#   nadata.dll | libnadata.so | libnadata.dylib   (+ nadata.lib on Windows)
#   nadata.h, nadata.hpp
#   nadata/        <- engine package (resolved relative to the library)
#   runtime/       <- embedded CPython (resolved relative to the library)
#
# The bundle is self-contained: do NOT copy files out of it, keep the
# directory intact and link against it in place.
#
# Usage (CMakeLists.txt):
#
#   include(${CMAKE_SOURCE_DIR}/third_party/nadata/nadataConfig.cmake)
#
#   add_executable(app main.c)
#   target_link_libraries(app PRIVATE nadata::nadata)
#
#   # POSIX: rpath entries pointing into the bundle are added
#   # automatically — nothing else needed.
#   # Windows (dev): run with the bundle directory on PATH, e.g. via
#   # the nadata-env.bat written by scripts/install.ps1, or deploy a
#   # self-contained copy next to the executable:
#   nadata_deploy_bundle(app)
#
# Requires CMake >= 3.11 (IMPORTED GLOBAL + INTERFACE_LINK_OPTIONS).

if(TARGET nadata::nadata)
    return()
endif()

get_filename_component(NADATA_ROOT "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

if(WIN32)
    set(NADATA_LIBRARY "${NADATA_ROOT}/nadata.dll")
    set(NADATA_IMPLIB "${NADATA_ROOT}/nadata.lib")
elseif(APPLE)
    set(NADATA_LIBRARY "${NADATA_ROOT}/libnadata.dylib")
else()
    set(NADATA_LIBRARY "${NADATA_ROOT}/libnadata.so")
endif()

if(NOT EXISTS "${NADATA_LIBRARY}")
    message(FATAL_ERROR "nadataConfig: ${NADATA_LIBRARY} not found — "
        "is this the staged bundle directory?")
endif()

add_library(nadata::nadata SHARED IMPORTED GLOBAL)

set_target_properties(nadata::nadata PROPERTIES
    IMPORTED_LOCATION "${NADATA_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NADATA_ROOT}")

if(WIN32)
    if(EXISTS "${NADATA_IMPLIB}")
        set_target_properties(nadata::nadata PROPERTIES
            IMPORTED_IMPLIB "${NADATA_IMPLIB}")
    else()
        message(WARNING "nadataConfig: nadata.lib missing — link will fail")
    endif()
else()
    # Resolve the shim and the embedded libpython from the bundle at
    # load time, for any executable location.
    set_target_properties(nadata::nadata PROPERTIES
        INTERFACE_LINK_OPTIONS
        "SHELL:-Wl,-rpath,${NADATA_ROOT};SHELL:-Wl,-rpath,${NADATA_ROOT}/runtime/lib")
endif()

# Optional: deploy a fully self-contained copy of the bundle next to
# TARGET's output (Windows-friendly: nadata.dll + python3xx.dll +
# nadata/ + runtime/ all land in the executable directory).
function(nadata_deploy_bundle TARGET)
    cmake_parse_arguments(NADATA_DEPLOY "" "DEST" "" ${ARGN})

    if(NOT NADATA_DEPLOY_DEST)
        set(NADATA_DEPLOY_DEST "$<TARGET_FILE_DIR:${TARGET}>")
    endif()

    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${NADATA_ROOT}" "${NADATA_DEPLOY_DEST}"
        COMMAND ${CMAKE_COMMAND} -E echo
            "nadata: deployed bundle to ${NADATA_DEPLOY_DEST}"
        COMMENT "Deploying the nadata bundle")
endfunction()

mark_as_advanced(NADATA_ROOT NADATA_LIBRARY NADATA_IMPLIB)
