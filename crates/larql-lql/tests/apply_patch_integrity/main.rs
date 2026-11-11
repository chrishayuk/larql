//! `APPLY PATCH` applies every operation or leaves the overlay as it was.
//!
//! Gate A5. A patch is refused whole, with an error, when it carries a
//! corrupt vector, a vector of the wrong width, a slot outside the index, or
//! the name of another model. Before this gate the VINDEX2 path discarded
//! the result of the checked apply and answered `Applied` for a patch it had
//! dropped, and neither backend checked the model, the width or the bounds.
//!
//! Every refusal below is checked the same way: the statement errors, and
//! what the session shows afterwards is identical to what it showed before.

mod support;
mod v2;
mod v3;
