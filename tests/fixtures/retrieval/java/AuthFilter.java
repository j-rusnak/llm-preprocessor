package com.example.preprocessor.security;

import java.io.IOException;
import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

public final class AuthFilter implements Filter {
    private final HmacVerifier verifier;

    public AuthFilter(HmacVerifier verifier) {
        this.verifier = verifier;
    }

    @Override
    public void doFilter(ServletRequest request,
                         ServletResponse response,
                         FilterChain chain) throws IOException, ServletException {
        HttpServletRequest http = (HttpServletRequest) request;
        String token = http.getHeader("X-Preprocessor-Authorization");
        String signature = http.getHeader("X-Preprocessor-Signature");
        String timestamp = http.getHeader("X-Preprocessor-Timestamp");
        if (!verifier.acceptsBearer(token) &&
            !verifier.acceptsHmac(signature, timestamp, http.getInputStream())) {
            ((HttpServletResponse) response).sendError(401);
            return;
        }
        chain.doFilter(request, response);
    }
}
