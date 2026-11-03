/* Link-time stand-ins for ggml symbols that ggml-quants.c references outside the Q8_0 reference encoder. None of them is
   reached by quantize_row_q8_0_ref; any call aborts, so a stand-in can never silently change a result. */
#include <stdlib.h>
void ggml_abort(void) { abort(); }
void ggml_row_size(void) { abort(); }
void ggml_type_name(void) { abort(); }
void ggml_type_size(void) { abort(); }
