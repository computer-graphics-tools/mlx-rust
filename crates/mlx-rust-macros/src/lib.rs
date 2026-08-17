//! Procedural macros for `mlx-rust`.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Attribute, FnArg, ItemFn, Pat, spanned::Spanned};

/// Generate a twin of a `*_device` function that supplies the default stream.
///
/// The function must be named `*_device` and take its stream last, named
/// `stream`; the twin drops both. Works on free functions and methods.
///
/// ```ignore
/// #[default_device]
/// pub fn quantize_device(w: &Array, stream: impl AsRef<Stream>) -> Result<Array> { .. }
///
/// // also generates:
/// pub fn quantize(w: &Array) -> Result<Array> {
///     Stream::with_default(|stream| quantize_device(w, stream))
/// }
/// ```
///
/// Strips the inert `#[optional]` parameter markers. No `optional` proc macro
/// exists to fall back on, so a stray marker is an error rather than a no-op.
#[proc_macro_attribute]
pub fn default_device(
    _attr: TokenStream,
    item: TokenStream,
) -> TokenStream {
    let mut func = match syn::parse::<ItemFn>(item) {
        Ok(func) => func,
        Err(err) => return err.to_compile_error().into(),
    };
    let twin = match build_twin(&func) {
        Ok(twin) => twin,
        Err(err) => return err.to_compile_error().into(),
    };
    strip_optional_markers(&mut func);
    quote! { #func #twin }.into()
}

fn build_twin(func: &ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let signature = &func.sig;

    let base_name = signature
        .ident
        .to_string()
        .strip_suffix("_device")
        .map(str::to_owned)
        .ok_or_else(|| {
            syn::Error::new(
                signature.ident.span(),
                "#[default_device] requires a name ending in `_device`",
            )
        })?;
    let twin_name = format_ident!("{}", base_name);
    let device_name = &signature.ident;

    let (receiver, rest) = split_receiver(signature);
    let (stream, inputs) = rest.split_last().ok_or_else(|| {
        syn::Error::new(
            signature.span(),
            "#[default_device] requires a trailing `stream` parameter",
        )
    })?;
    if param_name(stream)? != "stream" {
        return Err(syn::Error::new(
            stream.span(),
            "#[default_device] requires the last parameter to be named `stream`",
        ));
    }

    let params = inputs.iter().map(|arg| {
        let mut arg = (*arg).clone();
        if let FnArg::Typed(typed) = &mut arg {
            typed.attrs.clear();
        }
        arg
    });
    let forwarded = inputs
        .iter()
        .map(|arg| param_name(arg).map(|name| format_ident!("{}", name)))
        .collect::<syn::Result<Vec<_>>>()?;

    let call = match receiver {
        Some(_) => quote! { self.#device_name(#(#forwarded,)* stream) },
        None => quote! { #device_name(#(#forwarded,)* stream) },
    };
    let signature_params = receiver
        .map(|arg| quote! { #arg, })
        .into_iter()
        .chain(params.map(|arg| quote! { #arg, }));

    let vis = &func.vis;
    let attrs = forwarded_attrs(&func.attrs);
    let generics = &signature.generics;
    let where_clause = &signature.generics.where_clause;
    let output = &signature.output;

    Ok(quote! {
        #(#attrs)*
        /// Uses the default stream; see the `_device` form to choose one.
        #vis fn #twin_name #generics (#(#signature_params)*) #output
        #where_clause
        {
            crate::Stream::with_default(|stream| #call)
        }
    })
}

fn split_receiver(signature: &syn::Signature) -> (Option<&FnArg>, Vec<&FnArg>) {
    let mut params = signature.inputs.iter();
    match params.clone().next() {
        Some(receiver @ FnArg::Receiver(_)) => {
            params.next();
            (Some(receiver), params.collect())
        },
        _ => (None, params.collect()),
    }
}

fn param_name(arg: &FnArg) -> syn::Result<String> {
    match arg {
        FnArg::Typed(typed) => match &*typed.pat {
            Pat::Ident(ident) => Ok(ident.ident.to_string()),
            other => Err(syn::Error::new(
                other.span(),
                "#[default_device] requires plain identifier parameters",
            )),
        },
        FnArg::Receiver(_) => {
            Err(syn::Error::new(arg.span(), "unexpected receiver"))
        },
    }
}

/// The attributes that must apply to the twin as well.
fn forwarded_attrs(attrs: &[Attribute]) -> Vec<&Attribute> {
    attrs
        .iter()
        .filter(|attr| {
            attr.path().is_ident("doc")
                || attr.path().is_ident("cfg")
                || attr.path().is_ident("allow")
                || attr.path().is_ident("expect")
                || attr.path().is_ident("must_use")
        })
        .collect()
}

fn strip_optional_markers(func: &mut ItemFn) {
    for arg in &mut func.sig.inputs {
        if let FnArg::Typed(typed) = arg {
            typed.attrs.retain(|attr| !attr.path().is_ident("optional"));
        }
    }
}

/// Generate a `macro_rules!` call form with named optional arguments.
///
/// Required parameters stay positional; `#[optional]` ones become `name = value`
/// in any order and may be omitted:
///
/// ```ignore
/// quantized_matmul!(&x, &w, &scales, transpose = true)?
/// ```
///
/// Must sit above `#[default_device]` so it sees the markers and targets the
/// twin. The emitted `#[doc(hidden)]` args struct is what allows one arm to take
/// the names in any order: struct fields escape macro hygiene, where a `let`
/// binding would not.
#[proc_macro_attribute]
pub fn generate_macro(
    _attr: TokenStream,
    item: TokenStream,
) -> TokenStream {
    let func = match syn::parse::<ItemFn>(item) {
        Ok(func) => func,
        Err(err) => return err.to_compile_error().into(),
    };
    match build_named_macro(&func) {
        Ok(extra) => quote! { #func #extra }.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn build_named_macro(func: &ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let signature = &func.sig;
    let base_name = signature
        .ident
        .to_string()
        .strip_suffix("_device")
        .map(str::to_owned)
        .ok_or_else(|| {
            syn::Error::new(
                signature.ident.span(),
                "#[generate_macro] requires a name ending in `_device`",
            )
        })?;

    let (receiver, rest) = split_receiver(signature);
    if receiver.is_some() {
        return Err(syn::Error::new(
            signature.span(),
            "#[generate_macro] does not support methods",
        ));
    }
    let (_stream, inputs) = rest.split_last().ok_or_else(|| {
        syn::Error::new(
            signature.span(),
            "expected a trailing `stream` parameter",
        )
    })?;

    let mut required = Vec::new();
    let mut optional = Vec::new();
    for arg in inputs {
        let typed = match arg {
            FnArg::Typed(typed) => typed,
            FnArg::Receiver(_) => unreachable!("receiver split off above"),
        };
        let name = format_ident!("{}", param_name(arg)?);
        if typed.attrs.iter().any(|attr| attr.path().is_ident("optional")) {
            optional.push((name, option_inner(&typed.ty)?));
        } else {
            required.push(name);
        }
    }
    if optional.is_empty() {
        return Ok(quote! {});
    }

    let macro_name = format_ident!("{}", base_name);
    let fn_name = format_ident!("{}", base_name);
    let args_name = format_ident!("{}Args", upper_camel(&base_name));
    let lifetimes: Vec<_> = signature.generics.lifetimes().collect();
    let generics = if lifetimes.is_empty() {
        quote! {}
    } else {
        quote! { <#(#lifetimes),*> }
    };

    let fields = optional.iter().map(|(name, ty)| {
        quote! { pub #name: ::core::option::Option<#ty> }
    });
    let field_names: Vec<_> = optional.iter().map(|(name, _)| name).collect();

    // `quote!` claims `#` and `macro_rules!` claims `$`, so a metavariable is a
    // literal `$` plus an interpolated ident.
    let matchers = required.iter().map(|name| quote! { $#name:expr });
    let forwards = required.iter().map(|name| quote! { $#name });

    let optional_names = optional
        .iter()
        .map(|(name, _)| name.to_string())
        .collect::<Vec<_>>()
        .join("`, `");
    let macro_doc = format!(
        "Call [`{base_name}`] with named optional arguments, in any order: \
         `{optional_names}`.\n\nRequired arguments stay positional. \
         Exported at the crate root, so it is `crate::{base_name}!`."
    );

    Ok(quote! {
        #[doc(hidden)]
        #[derive(Default)]
        pub struct #args_name #generics {
            #(#fields,)*
        }

        #[doc = #macro_doc]
        #[macro_export]
        macro_rules! #macro_name {
            (
                #(#matchers),*
                $(, $__field:ident = $__value:expr)* $(,)?
            ) => {{
                let mut args = $crate::ops::#args_name::default();
                $( args.$__field = ::core::option::Option::Some($__value); )*
                $crate::ops::#fn_name(
                    #(#forwards,)*
                    #(args.#field_names,)*
                )
            }};
        }
    })
}

/// Pull `T` out of `impl Into<Option<T>>`.
fn option_inner(ty: &syn::Type) -> syn::Result<syn::Type> {
    let err = || {
        syn::Error::new(
            ty.span(),
            "#[optional] parameters must be `impl Into<Option<T>>`",
        )
    };
    let syn::Type::ImplTrait(impl_trait) = ty else {
        return Err(err());
    };
    for bound in &impl_trait.bounds {
        let syn::TypeParamBound::Trait(bound) = bound else {
            continue;
        };
        let into_segment = bound.path.segments.last().ok_or_else(err)?;
        if into_segment.ident != "Into" {
            continue;
        }
        let syn::PathArguments::AngleBracketed(into_args) =
            &into_segment.arguments
        else {
            continue;
        };
        let Some(syn::GenericArgument::Type(syn::Type::Path(option_path))) =
            into_args.args.first()
        else {
            continue;
        };
        let option_segment =
            option_path.path.segments.last().ok_or_else(err)?;
        if option_segment.ident != "Option" {
            continue;
        }
        let syn::PathArguments::AngleBracketed(option_args) =
            &option_segment.arguments
        else {
            continue;
        };
        if let Some(syn::GenericArgument::Type(inner_type)) =
            option_args.args.first()
        {
            return Ok(inner_type.clone());
        }
    }
    Err(err())
}

fn upper_camel(snake: &str) -> String {
    snake
        .split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => {
                    first.to_uppercase().collect::<String>() + chars.as_str()
                },
                None => String::new(),
            }
        })
        .collect()
}
