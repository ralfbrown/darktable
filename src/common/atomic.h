/*
    This file is part of darktable,
    Copyright (C) 2020 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#if !defined(__STDC_NO_ATOMICS__) && defined(DT_USE_ATOMICS)

#include <stdatomic.h>

typedef atomic_int dt_atomic_int;
inline void dt_atomic_set_int(atomic_int *var, int value) { atomic_store(var,value); }
inline int dt_atomic_get_int(atomic_int *var) { return atomic_load(var); }

#else // we don't have or aren't supposed to use atomics

// fall back to using a mutex for synchronization
#include <pthread.h>

extern pthread_mutex_t dt_atom_mutex;

typedef int dt_atomic_int;
inline void dt_atomic_set_int(dt_atomic_int *var, int value)
{
  pthread_mutex_lock(&dt_atom_mutex);
  *var = value;
  pthread_mutex_unlock(&dt_atom_mutex);
}

inline int dt_atomic_get_int(const dt_atomic_int *const var)
{
  pthread_mutex_lock(&dt_atom_mutex);
  int value = *var;
  pthread_mutex_unlock(&dt_atom_mutex);
  return value;
}

#endif // __STDC_NO_ATOMICS__
